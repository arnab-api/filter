import logging
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from tqdm.auto import tqdm

from src.functional import (
    get_all_module_states,
    get_module_nnsight,
    interpret_logits,
    predict_next_token,
)
from src.models import ModelandTokenizer
from src.tokens import align_patching_positions
from src.utils.typing import PathLike, PredictedToken, TokenizerOutput

logger = logging.getLogger(__name__)


@torch.inference_mode()
def patched_run(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    states: dict[tuple[str, int], torch.Tensor],
) -> torch.Tensor:
    # os.environ["TORCH_LOGS"] = "not_implemented"

    with mt.trace(inputs, scan=False) as trace:
        for location in states:
            layer_name, token_idx = location
            module = get_module_nnsight(mt, layer_name)
            current_states = (
                module.output if ("mlp" in layer_name) else module.output[0]
            )
            current_states[0, token_idx, :] = states[location]
        logits = mt.output.logits[0][-1].save()
    return logits


def get_window(layer_name_format, idx, window_size, n_layer):
    return [
        layer_name_format.format(i)
        for i in range(
            max(0, idx - window_size // 2), min(n_layer - 1, idx + window_size // 2) + 1
        )
    ]


def rank_reward(rank, k=20):
    """
    will return a positive reward if rank is less then 20 (negative log curve)
    will clip the reward to 0 if rank is >= 20
    """
    assert rank >= 1, "rank must be >= 1"
    assert k > 1, "k must be > 1"
    buffer = np.log(k)
    y = (-np.log(rank) + buffer) / buffer
    y = np.clip(y, 0, None)
    return y


def get_score(
    logits: torch.Tensor,
    token_id: int | list[int],
    metric: Literal["logit", "prob", "log_norm", "log_rank_inv"] = "logit",
    return_individual_scores: bool = False,
    k: int | None = 20,
) -> Union[float, torch.Tensor]:
    token_id = [token_id] if isinstance(token_id, int) else token_id
    logits = logits.squeeze()
    logits = logits.softmax(dim=-1) if metric == "prob" else logits
    if metric == "log_norm":
        assert k is not None, "k must be provided for log_norm"
        denom = logits.topk(k=k, dim=-1).values.mean(dim=-1)
        # logger.debug(f"{logits.shape} | {logits[token_id]=} | {denom=}")
        # logits = logits / denom #! ratio of logits is a weird metric
        logits = logits - denom  #! difference probably makes more sense (?)
    elif metric == "log_rank_inv":
        assert k is not None, "k must be provided for log_rank_inv"
        rank = logits.argsort(dim=-1, descending=True) + 1
        inv_reward = [rank_reward(rank[t], k=k) for t in token_id]
        inv_reward = sum(inv_reward) / len(inv_reward)
        return inv_reward
    score = logits[token_id].mean().item()
    if not return_individual_scores:
        return score
    individual_scores = {t: logits[t].item() for t in token_id}
    return score, individual_scores


@torch.inference_mode()
def calculate_indirect_effects(
    mt: ModelandTokenizer,
    locations: list[tuple[int, int]],  # layer_idx, token_idx
    clean_input: TokenizerOutput,
    patch_states: dict[
        tuple[str, int], torch.Tensor
    ],  # expects the states to be in clean_states
    patch_ans_t: int,
    layer_name_format: str,
    window_size: int = 1,
    metric: Literal["logit", "prob", "log_norm"] = "prob",
) -> dict[tuple[str, int], float]:
    # logger.debug(
    #     f"===> {len(locations)=} | {window_size=} | {mt.tokenizer.decode(patch_ans_t)}{patch_ans_t}"
    # )
    indirect_effects = {loc: -1 for loc in locations}
    for loc in tqdm(locations):
        layer_names = get_window(layer_name_format, loc[0], window_size, mt.n_layer)
        token_idx = loc[1]
        states = {(l, token_idx): patch_states[(l, token_idx)] for l in layer_names}
        affected_logits = patched_run(
            mt=mt,
            inputs=clean_input,
            states=states,
        )
        # value = (
        #     affected_logits.softmax(dim=-1)[patch_ans_t].item()
        #     if metric == "prob"
        #     else affected_logits.squeeze()[patch_ans_t].item()
        # )
        indirect_effects[loc] = get_score(
            logits=affected_logits,
            token_id=patch_ans_t,
            metric=metric,
            return_individual_scores=False,
        )
        # print(loc, indirect_effects[loc])
    return indirect_effects


@dataclass
class CausalTracingResult(DataClassJsonMixin):
    patch_input_toks: list[str]
    corrupt_input_toks: list[str]
    trace_start_idx: int
    answer: list[PredictedToken]
    low_score: float
    base_score: float
    indirect_effects: torch.Tensor
    subj_range: Optional[tuple[int, int]]
    normalized: bool
    kind: Literal["residual", "mlp", "attention"] = "residual"
    window: int = 1
    metric: Literal["logit", "prob"] = "prob"

    def from_npz(file: np.lib.npyio.NpzFile | PathLike):
        if isinstance(file, PathLike):
            file = np.load(file, allow_pickle=True)

        return CausalTracingResult(
            patch_input_toks=file["patch_input_toks"].tolist(),
            corrupt_input_toks=file["corrupt_input_toks"].tolist(),
            trace_start_idx=file["trace_start_idx"].item(),
            answer=file["answer"].tolist(),
            subj_range=file["subj_range"].tolist(),
            low_score=file["low_score"].item(),
            base_score=file["base_score"].item(),
            indirect_effects=torch.tensor(file["indirect_effects"]),
            normalized=file["normalized"].item(),
            kind=file["kind"].item(),
            window=file["window"].item(),
            metric=file["metric"].item(),
        )


@torch.inference_mode()
def trace_important_states(
    mt: ModelandTokenizer,
    prompt_template: str,
    clean_subj: str,
    patched_subj: str,
    clean_input: Optional[TokenizerOutput] = None,
    patched_input: Optional[TokenizerOutput] = None,
    kind: Literal["residual", "mlp", "attention"] = "residual",
    window_size: int = 1,
    normalize=True,
    trace_start_marker: Optional[str] = None,
    metric: Literal["logit", "prob", "log_norm"] = "prob",
    ans_tokens: Optional[list[int] | int] = None,
) -> CausalTracingResult:
    aligned = align_patching_positions(
        mt=mt,
        prompt_template=prompt_template,
        clean_subj=clean_subj,
        patched_subj=patched_subj,
        clean_input=clean_input,
        patched_input=patched_input,
        trace_start_marker=trace_start_marker,
    )
    clean_input = aligned["clean_input"]
    patched_input = aligned["patched_input"]
    subj_range = aligned["subj_range"]
    trace_start_idx = aligned["trace_start_idx"]

    print(f"===> {trace_start_idx=}")

    if trace_start_marker is None:
        trace_start_idx = 0
        if (
            clean_input.input_ids[0][0]
            == patched_input.input_ids[0][0]
            == mt.tokenizer.pad_token_id
        ):
            trace_start_idx = 1

    # base run with the patched subject
    patched_states = get_all_module_states(mt=mt, input=patched_input, kind=kind)

    if ans_tokens is None:
        # interested answer
        logits = patched_run(mt=mt, inputs=patched_input, states={})
        answer = interpret_logits(tokenizer=mt.tokenizer, logits=logits)[0]
        base_score = get_score(logits=logits, token_id=answer.token_id, metric=metric)
        logger.debug(f"{answer=}")

        # clean run
        clean_logits = patched_run(mt=mt, inputs=clean_input, states={})
        clean_answer, track_ans = interpret_logits(
            tokenizer=mt.tokenizer,
            logits=clean_logits,
            interested_tokens=[answer.token_id],
        )
        clean_answer = clean_answer[0]
        track_ans = track_ans[answer.token_id][1]

        logger.debug(f"{clean_answer=}")
        logger.debug(f"{track_ans=}")
        assert (
            answer.token != clean_answer.token
        ), "Answers in the clean and corrupt runs are the same"

        low_score = get_score(
            logits=clean_logits, token_id=answer.token_id, metric=metric
        )

        ans_tokens = [answer.token_id]
        answer = [answer]
    else:
        base_score, base_indv_scores = get_score(
            logits=patched_run(
                mt=mt,
                inputs=patched_input,
                states={},  # don't patch anything
            ),
            token_id=ans_tokens,
            metric=metric,
            return_individual_scores=True,
        )
        answer = []
        logger.debug(f"{base_score=} | {base_indv_scores=}")

        for tok in base_indv_scores:
            pred = PredictedToken(
                token=mt.tokenizer.decode(tok),
                token_id=tok,
            )
            if metric in ["logit", "prob"]:
                setattr(pred, metric, base_indv_scores[tok])
            else:
                pred.metadata = {metric: base_indv_scores[tok]}

            answer.append(pred)

        low_score, low_indv_scores = get_score(
            logits=patched_run(
                mt=mt,
                inputs=clean_input,
                states={},  # don't patch anything
            ),
            token_id=ans_tokens,
            metric=metric,
            return_individual_scores=True,
        )
        logger.debug(f"{low_score=} | {low_indv_scores=}")

    assert (
        low_score < base_score
    ), f"{low_score=} | {base_score=} >> low_score must be less than base_score"
    logger.debug(f"{base_score=} | {low_score=}")

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError("kind must be one of 'residual', 'mlp', 'attention'")

    logger.debug(f"---------- tracing important states | {kind=} ----------")
    # calculate indirect effects in the patched run
    locations = [
        (layer_idx, token_idx)
        for layer_idx in range(mt.n_layer)
        for token_idx in range(trace_start_idx, clean_input.input_ids.size(1))
    ]
    indirect_effects = calculate_indirect_effects(
        mt=mt,
        locations=locations,
        clean_input=clean_input,
        patch_states=patched_states,
        patch_ans_t=ans_tokens,
        layer_name_format=layer_name_format,
        window_size=window_size,
        metric=metric,
    )

    indirect_effect_matrix = []
    for token_idx in range(trace_start_idx, clean_input.input_ids.size(1)):
        indirect_effect_matrix.append(
            [
                indirect_effects[(layer_idx, token_idx)]
                for layer_idx in range(mt.n_layer)
            ]
        )

    indirect_effect_matrix = torch.tensor(indirect_effect_matrix)
    if normalize:
        logger.info(f"{base_score=} | {low_score=}")
        indirect_effect_matrix = (indirect_effect_matrix - low_score) / (
            base_score - low_score
        )

    return CausalTracingResult(
        patch_input_toks=[
            mt.tokenizer.decode(tok) for tok in patched_input.input_ids[0]
        ],
        corrupt_input_toks=[
            mt.tokenizer.decode(tok) for tok in clean_input.input_ids[0]
        ],
        trace_start_idx=trace_start_idx,
        answer=answer,
        subj_range=subj_range,
        low_score=low_score,
        base_score=base_score,
        indirect_effects=indirect_effect_matrix,
        normalized=normalize,
        kind=kind,
        window=window_size,
        metric=metric,
    )


# @torch.inference_mode()
# def trace_important_states_RAG(
#     mt: ModelandTokenizer,
#     clean_query: InContextQuery,
#     corrupt_query: InContextQuery,
#     kind: Literal["residual", "mlp", "attention"] = "residual",
#     trace_token_strategy: Literal["subj_query", "all"] = "subj_query",
#     window_size: int = 1,
#     normalize=True,
# ) -> CausalTracingResult:
#     assert (
#         clean_query.template == corrupt_query.template
#     ), "Queries do not have the same template"

#     clean_inputs = prepare_input(
#         prompts=clean_query.query, tokenizer=mt, return_offset_mapping=True
#     )
#     corrupt_inputs = prepare_input(
#         prompts=corrupt_query.query, tokenizer=mt, return_offset_mapping=True
#     )

#     if trace_token_strategy == "subj_query":
#         clean_subj_range = find_token_range(
#             string=clean_query.query,
#             substring=clean_query.subject,
#             tokenizer=mt.tokenizer,
#             occurrence=-1,
#             offset_mapping=clean_inputs["offset_mapping"][0],
#         )
#         corrupt_subj_range = find_token_range(
#             string=corrupt_query.query,
#             substring=corrupt_query.subject,
#             tokenizer=mt.tokenizer,
#             occurrence=-1,
#             offset_mapping=corrupt_inputs["offset_mapping"][0],
#         )
#         logger.debug(f"{clean_subj_range=} | {corrupt_subj_range=}")

#         # always insert 1 padding token
#         subj_end = max(clean_subj_range[1], corrupt_subj_range[1]) + 1
#         logger.debug(f"setting {subj_end=}")

#         clean_inputs = insert_padding_before_subj(
#             clean_inputs,
#             clean_subj_range,
#             subj_end,
#             pad_id=mt.tokenizer.pad_token_id,
#         )
#         corrupt_inputs = insert_padding_before_subj(
#             corrupt_inputs,
#             corrupt_subj_range,
#             subj_end,
#             pad_id=mt.tokenizer.pad_token_id,
#         )

#         clean_shift = subj_end - clean_subj_range[1]
#         clean_subj_range = (clean_subj_range[0] + clean_shift, subj_end)

#         corrupt_shift = subj_end - corrupt_subj_range[1]
#         corrupt_subj_range = (corrupt_subj_range[0] + corrupt_shift, subj_end)

#         logger.debug(f"<shifted> {clean_subj_range=} | {corrupt_subj_range=}")

#     elif trace_token_strategy == "all":
#         clean_subj_ranges = [
#             find_token_range(
#                 string=clean_query.query,
#                 substring=clean_query.subject,
#                 tokenizer=mt.tokenizer,
#                 occurrence=order,
#                 offset_mapping=clean_inputs["offset_mapping"][0],
#             )
#             for order in [0, -1]
#         ]

#         corrupt_subj_ranges = [
#             find_token_range(
#                 string=corrupt_query.query,
#                 substring=corrupt_query.subject,
#                 tokenizer=mt.tokenizer,
#                 occurrence=order,
#                 offset_mapping=corrupt_inputs["offset_mapping"][0],
#             )
#             for order in [0, -1]
#         ]

#         clean_cofa_range = find_token_range(
#             string=clean_query.query,
#             substring=guess_subject(clean_query.cf_description),
#             tokenizer=mt.tokenizer,
#             occurrence=-1,
#             offset_mapping=clean_inputs["offset_mapping"][0],
#         )

#         corrupt_cofa_range = find_token_range(
#             string=corrupt_query.query,
#             substring=guess_subject(corrupt_query.cf_description),
#             tokenizer=mt.tokenizer,
#             occurrence=-1,
#             offset_mapping=corrupt_inputs["offset_mapping"][0],
#         )

#         # align the subjects in the context
#         subj_end_in_context = (
#             max(clean_subj_ranges[0][1], corrupt_subj_ranges[0][1]) + 1
#         )
#         clean_inputs = insert_padding_before_subj(
#             clean_inputs,
#             clean_subj_ranges[0],
#             subj_end_in_context,
#             pad_id=mt.tokenizer.pad_token_id,
#         )
#         corrupt_inputs = insert_padding_before_subj(
#             corrupt_inputs,
#             corrupt_subj_ranges[0],
#             subj_end_in_context,
#             pad_id=mt.tokenizer.pad_token_id,
#         )

#         n_clean_pads = subj_end_in_context - clean_subj_ranges[0][1]
#         clean_subj_ranges[1] = (
#             clean_subj_ranges[1][0] + n_clean_pads,
#             clean_subj_ranges[1][1] + n_clean_pads,
#         )
#         clean_cofa_range = (
#             clean_cofa_range[0] + n_clean_pads,
#             clean_cofa_range[1] + n_clean_pads,
#         )

#         n_corrupt_pads = subj_end_in_context - corrupt_subj_ranges[0][1]
#         corrupt_subj_ranges[1] = (
#             corrupt_subj_ranges[1][0] + n_corrupt_pads,
#             corrupt_subj_ranges[1][1] + n_corrupt_pads,
#         )
#         corrupt_cofa_range = (
#             corrupt_cofa_range[0] + n_corrupt_pads,
#             corrupt_cofa_range[1] + n_corrupt_pads,
#         )

#         # align the counterfactuals in the context
#         cofa_ends_in_context = max(clean_cofa_range[1], corrupt_cofa_range[1]) + 1
#         clean_inputs = insert_padding_before_subj(
#             clean_inputs,
#             clean_cofa_range,
#             cofa_ends_in_context,
#             pad_id=mt.tokenizer.pad_token_id,
#         )
#         corrupt_inputs = insert_padding_before_subj(
#             corrupt_inputs,
#             corrupt_cofa_range,
#             cofa_ends_in_context,
#             pad_id=mt.tokenizer.pad_token_id,
#         )

#         n_clean_pads = cofa_ends_in_context - clean_cofa_range[1]
#         clean_subj_ranges[1] = (
#             clean_subj_ranges[1][0] + n_clean_pads,
#             clean_subj_ranges[1][1] + n_clean_pads,
#         )
#         n_corrupt_pads = cofa_ends_in_context - corrupt_cofa_range[1]
#         corrupt_subj_ranges[1] = (
#             corrupt_subj_ranges[1][0] + n_corrupt_pads,
#             corrupt_subj_ranges[1][1] + n_corrupt_pads,
#         )

#         # align the subjects in the query
#         subj_ends_in_query = max(clean_subj_ranges[1][1], corrupt_subj_ranges[1][1]) + 1
#         clean_inputs = insert_padding_before_subj(
#             clean_inputs,
#             clean_subj_ranges[1],
#             subj_ends_in_query,
#             pad_id=mt.tokenizer.pad_token_id,
#         )
#         corrupt_inputs = insert_padding_before_subj(
#             corrupt_inputs,
#             corrupt_subj_ranges[1],
#             subj_ends_in_query,
#             pad_id=mt.tokenizer.pad_token_id,
#         )

#     else:
#         raise ValueError("trace_token_strategy must be one of 'subj_query', 'all'")

#     for idx, (t1, a1, t2, a2) in enumerate(
#         zip(
#             clean_inputs.input_ids[0],
#             clean_inputs.attention_mask[0],
#             corrupt_inputs.input_ids[0],
#             corrupt_inputs.attention_mask[0],
#         )
#     ):
#         logger.debug(
#             f"{idx=} =>  [{a1}] {mt.tokenizer.decode(t1)} || [{a2}] {mt.tokenizer.decode(t2)}"
#         )

#     # trace start idx
#     if trace_token_strategy == "subj_query":
#         trace_start_idx = min(clean_subj_range[0], corrupt_subj_range[0])
#     elif trace_token_strategy == "all":
#         trace_start_idx = 1

#     # clean run
#     clean_states = get_all_module_states(mt=mt, input=clean_inputs, kind=kind)
#     answer = predict_next_token(mt=mt, inputs=clean_inputs, k=1)[0][0]
#     base_probability = answer.prob
#     logger.debug(f"{answer=}")

#     # corrupted run
#     # corrupt_states = get_all_module_states(mt=mt, input=corrupt_inputs, kind=kind)
#     corrupt_answer, track_ans = predict_next_token(
#         mt=mt, inputs=corrupt_inputs, k=1, token_of_interest=answer.token
#     )
#     corrupt_answer = corrupt_answer[0][0]
#     corrupt_probability = track_ans[0][1].prob
#     logger.debug(f"{corrupt_answer=}")
#     logger.debug(f"{track_ans=}")

#     logger.debug("---------- tracing important states ----------")

#     assert (
#         answer.token != corrupt_answer.token
#     ), "Answers in the clean and corrupt runs are the same"

#     layer_name_format = None
#     if kind == "residual":
#         layer_name_format = mt.layer_name_format
#     elif kind == "mlp":
#         layer_name_format = mt.mlp_module_name_format
#     elif kind == "attention":
#         layer_name_format = mt.attn_module_name_format
#     else:
#         raise ValueError(f"kind must be one of 'residual', 'mlp', 'attention'")

#     # calculate indirect effects in the patched run
#     locations = [
#         (layer_idx, token_idx)
#         for layer_idx in range(mt.n_layer)
#         for token_idx in range(trace_start_idx, clean_inputs.input_ids.size(1))
#     ]
#     indirect_effects = calculate_indirect_effects(
#         mt=mt,
#         locations=locations,
#         corrupted_input=corrupt_inputs,
#         patch_states=clean_states,
#         patch_ans_t=answer.token_id,
#         layer_name_format=layer_name_format,
#         window_size=window_size,
#         kind=kind,
#     )

#     indirect_effect_matrix = []
#     for token_idx in range(trace_start_idx, clean_inputs.input_ids.size(1)):
#         indirect_effect_matrix.append(
#             [
#                 indirect_effects[(layer_idx, token_idx)]
#                 for layer_idx in range(mt.n_layer)
#             ]
#         )

#     indirect_effect_matrix = torch.tensor(indirect_effect_matrix)
#     if normalize:
#         indirect_effect_matrix = (indirect_effect_matrix - corrupt_probability) / (
#             base_probability - corrupt_probability
#         )

#     return CausalTracingResult(
#         patch_input_toks=[
#             mt.tokenizer.decode(tok) for tok in clean_inputs.input_ids[0]
#         ],
#         corrupt_input_toks=[
#             mt.tokenizer.decode(tok) for tok in corrupt_inputs.input_ids[0]
#         ],
#         trace_start_idx=trace_start_idx,
#         answer=answer,
#         low_score=corrupt_probability,
#         indirect_effects=indirect_effect_matrix,
#         normalized=normalize,
#         kind=kind,
#         window=window_size,
#     )
