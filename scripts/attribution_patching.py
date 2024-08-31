import argparse
import json
import logging
import os
import types
from dataclasses import dataclass
from typing import Literal, Optional

import baukit  # type: ignore
import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from tqdm import tqdm

from src.dataset import (
    BridgeDataset,
    BridgeRelation,
    BridgeSample,
    load_bridge_relation,
)
from src.functional import (
    PatchSpec,
    find_token_range,
    free_gpu_cache,
    get_hs,
    get_module_nnsight,
    guess_subject,
    predict_next_token,
    prepare_input,
    untuple,
)
from src.hooking.llama_attention import AttentionEdge, LlamaAttentionPatcher
from src.models import ModelandTokenizer, prepare_input
from src.trace import insert_padding_before_subj
from src.utils import env_utils, logging_utils
from src.utils.typing import PredictedToken, TokenizerOutput

logger = logging.getLogger(__name__)


#! the clean here actually stands for **corrupt* in the causal tracing
def attribution_patching(
    mt: ModelandTokenizer,
    clean_inputs: TokenizerOutput,
    patches: PatchSpec | list[PatchSpec],
    interested_locations: list[tuple[str, int]],
    ans_token_idx: int,
    metric: Literal["logit", "proba"] = "proba",
    resolution: int = 10,
    intermediate_point_sample: Literal["linear", "non-linear"] = "non-linear",
) -> dict[tuple[str, int], float]:

    if "offset_mapping" in clean_inputs:
        clean_inputs.pop("offset_mapping")
    if isinstance(patches, PatchSpec):
        patches = [patches]

    clean_states = get_hs(
        mt=mt, input=clean_inputs, locations=interested_locations, return_dict=True
    )
    patched_states = get_hs(
        mt=mt,
        input=clean_inputs,
        locations=interested_locations,
        patches=patches,
        return_dict=True,
    )
    free_gpu_cache()

    scan = True

    @dataclass
    class IGCache:
        h: torch.Tensor
        grad: torch.Tensor

        # def __post_init__(self):
        #     assert self.h.shape == self.grad.shape

    def is_an_attn_head(module_name) -> bool | tuple[int, int]:
        attn_id = mt.attn_module_name_format.split(".")[-1]
        if attn_id not in module_name:
            return False
        if module_name.endswith(attn_id):
            return False

        head_id = module_name.split(".")[-1]
        layer_id = ".".join(module_name.split(".")[:-1])

        return layer_id, int(head_id)

    approx_IE = {loc: [] for loc in interested_locations}
    alphas = torch.linspace(0, 1, resolution + 1)

    for a_idx in tqdm(range(len(alphas) - 1)):
        alpha = alphas[a_idx]

        with mt.trace(clean_inputs, scan=False, validate=False) as tracer:
            # patching
            for patch in patches:
                module_name, tok_idx = patch.location
                patch_module = get_module_nnsight(mt, module_name)

                assert (
                    isinstance(patch.clean, torch.Tensor)
                    and patch.clean.shape == patch.patch.shape
                )
                mid_point = (1 - alpha) * patch.clean + alpha * patch.patch
                mid_point = (
                    mid_point.to(mt.device)
                    if not mid_point.device == mt.device
                    else mid_point
                )
                mid_point.retain_grad = True

                patch_module.output[0, tok_idx, :] = mid_point

            # cache the interested hidden states
            for loc in interested_locations:
                module_name, tok_idx = loc

                if is_an_attn_head(module_name) == False:
                    module = get_module_nnsight(mt, module_name)
                    # tracer.apply(logger.debug, f"{module_name} => {module.input.shape}")
                    cur_output = (
                        module.output.save()
                        if ("mlp" in module_name or module_name == mt.embedder_name)
                        else module.output[0].save()
                    )  #! nnsight quirk => to get the grad of a reference tensor, you can't index it
                    # if "mlp" in module_name:
                    #     cur_output = module.input[0][0].save()
                    # elif "attn" in module_name:
                    #     cur_output = module.input[1]["hidden_states"].save()
                    # else:
                    #     cur_output = module.input[0][0].save()

                    # tracer.apply(logger.debug, cur_output.shape)
                    approx_IE[loc].append(
                        IGCache(
                            h=cur_output[0, tok_idx, :].save(),
                            grad=cur_output.grad[0, tok_idx, :].save(),
                        )
                    )
                else:
                    attn_module_name, head_idx = is_an_attn_head(module_name)
                    o_proj_name = attn_module_name + ".o_proj"
                    head_dim = mt.n_embd // mt.model.config.num_attention_heads
                    o_proj = get_module_nnsight(mt, o_proj_name)
                    cur_input = o_proj.input[0][0][
                        :, :, head_idx * head_dim : (head_idx + 1) * head_dim
                    ].save()
                    # tracer.apply(logger.debug, cur_input.size())

                    approx_IE[loc].append(
                        IGCache(
                            h=cur_input[0, tok_idx, :].save(),
                            grad=o_proj.input[0][0]
                            .grad[
                                0,
                                tok_idx,
                                head_idx * head_dim : (head_idx + 1) * head_dim,
                            ]
                            .save(),
                        )
                    )

            #! nnsight quirk => backward() has to be called later than grad.save() to populate the proxies
            if metric == "logit":
                v = mt.output.logits[0][-1][ans_token_idx]
            elif metric == "proba":
                v = mt.output.logits[0][-1].softmax(dim=-1)[ans_token_idx]
            else:
                raise ValueError(f"unknown {metric=}")
            v.backward()

        mt._model.zero_grad()
        scan = False
        free_gpu_cache()

    grads = {
        loc: sum([ig.grad for ig in approx_IE[loc]]) / resolution
        for loc in interested_locations
    }

    if intermediate_point_sample == "linear":
        approx_IE = {
            loc: torch.dot(grad, patched_states[loc] - clean_states[loc]).item()
            for loc, grad in grads.items()
        }
    elif intermediate_point_sample == "non-linear":
        for loc in interested_locations:
            ie_approx = 0
            for a_idx in range(len(alphas) - 1):
                grad = approx_IE[loc][a_idx].grad
                h_0 = approx_IE[loc][a_idx].h
                h_next = (
                    approx_IE[loc][a_idx + 1].h
                    if a_idx < len(alphas) - 2
                    else patched_states[loc]
                )

                ie_approx += torch.dot(grad, h_next - h_0).item()

            approx_IE[loc] = ie_approx
    else:
        raise ValueError(f"unknown {intermediate_point_sample=}")

    return approx_IE


def sample_icq_with_different_bridge(
    icq_examples: list[BridgeSample], patch_icq: BridgeSample
) -> int:
    indices = np.random.permutation(len(icq_examples))
    for idx in indices:
        if icq_examples[idx].bridge != patch_icq.bridge:
            return idx
    raise ValueError("No sample found with a different bridge")


def process_inputs(
    mt: ModelandTokenizer,
    clean_icq: str,
    clean_entity_pair: tuple[str, str],
    patch_icq: str,
    patch_entity_pair: tuple[str, str],
) -> tuple[TokenizerOutput, TokenizerOutput, int]:
    clean_inputs = prepare_input(
        prompts=clean_icq[0], tokenizer=mt, return_offsets_mapping=True
    )

    clean_offsets = clean_inputs.pop("offset_mapping")[0]
    clean_subj_ranges = [
        find_token_range(
            string=clean_icq[0],
            substring=subj,
            tokenizer=mt.tokenizer,
            offset_mapping=clean_offsets,
            occurrence=-1,
        )
        for subj in clean_entity_pair
    ]
    logger.debug(f"{clean_subj_ranges=}")
    for t in range(*clean_subj_ranges[0]):
        logger.debug(f"{t=} | {mt.tokenizer.decode(clean_inputs['input_ids'][0][t])}")
    logger.debug(f"{'-'*50}")
    for t in range(*clean_subj_ranges[1]):
        logger.debug(f"{t=} | {mt.tokenizer.decode(clean_inputs['input_ids'][0][t])}")
    logger.debug(f"{'='*50}")

    patch_inputs = prepare_input(
        prompts=patch_icq[0], tokenizer=mt, return_offsets_mapping=True
    )
    patch_offsets = patch_inputs.pop("offset_mapping")[0]
    patch_subj_ranges = [
        find_token_range(
            string=patch_icq[0],
            substring=subj,
            tokenizer=mt.tokenizer,
            offset_mapping=patch_offsets,
            occurance=-1,
        )
        for subj in patch_entity_pair
    ]
    logger.debug(f"{patch_subj_ranges=}")
    for t in range(*patch_subj_ranges[0]):
        logger.debug(f"{t=} | {mt.tokenizer.decode(patch_inputs['input_ids'][0][t])}")
    logger.debug("-" * 50)
    for t in range(*patch_subj_ranges[1]):
        logger.debug(f"{t=} | {mt.tokenizer.decode(patch_inputs['input_ids'][0][t])}")

    logger.debug(f"{'+'*50}")

    assert clean_subj_ranges[0][0] == patch_subj_ranges[0][0]
    subj_1_range = (
        clean_subj_ranges[0][0],
        max(clean_subj_ranges[0][1], patch_subj_ranges[0][1]),
    )
    clean_inputs = insert_padding_before_subj(
        clean_inputs,
        clean_subj_ranges[0],
        subj_1_range[1],
        pad_id=mt.tokenizer.pad_token_id,
        fill_attn_mask=True,
    )
    patch_inputs = insert_padding_before_subj(
        patch_inputs,
        patch_subj_ranges[0],
        subj_1_range[1],
        pad_id=mt.tokenizer.pad_token_id,
        fill_attn_mask=True,
    )

    clean_subj2_shift = subj_1_range[1] - clean_subj_ranges[0][1]
    clean_subj_ranges[1] = (
        clean_subj_ranges[1][0] + clean_subj2_shift,
        clean_subj_ranges[1][1] + clean_subj2_shift,
    )
    patch_subj2_shift = subj_1_range[1] - patch_subj_ranges[0][1]
    patch_subj_ranges[1] = (
        patch_subj_ranges[1][0] + patch_subj2_shift,
        patch_subj_ranges[1][1] + patch_subj2_shift,
    )

    subj_2_range = (
        max(clean_subj_ranges[1][0], patch_subj_ranges[1][0]),
        max(clean_subj_ranges[1][1], patch_subj_ranges[1][1]),
    )
    clean_inputs = insert_padding_before_subj(
        clean_inputs,
        clean_subj_ranges[1],
        subj_2_range[1],
        pad_id=mt.tokenizer.pad_token_id,
        fill_attn_mask=True,
    )
    patch_inputs = insert_padding_before_subj(
        patch_inputs,
        patch_subj_ranges[1],
        subj_2_range[1],
        pad_id=mt.tokenizer.pad_token_id,
        fill_attn_mask=True,
    )

    # for idx, (t1, a1, t2, a2) in enumerate(zip(
    #     clean_inputs.input_ids[0], clean_inputs.attention_mask[0],
    #     patch_inputs.input_ids[0], patch_inputs.attention_mask[0],
    # )):
    #     is_subj = idx in range(subj_1_range[0], subj_1_range[1]) or idx in range(subj_2_range[0], subj_2_range[1])
    #     append = "*" if is_subj else ""
    #     print(f"{idx=} >> [{a1}] {mt.tokenizer.decode(t1)}{append} | [{a2}] {mt.tokenizer.decode(t2)}{append}")

    logger.debug(f"{subj_1_range=}")
    for i in range(*subj_1_range):
        logger.debug(
            f"{i=} | {mt.tokenizer.decode(clean_inputs['input_ids'][0][i])} <> {mt.tokenizer.decode(patch_inputs['input_ids'][0][i])}"
        )
    logger.debug(f"{'-'*50}")
    logger.debug(f"{subj_2_range=}")
    for i in range(*subj_2_range):
        logger.debug(
            f"{i=} | [{clean_inputs['attention_mask'][0][i]}]{mt.tokenizer.decode(clean_inputs['input_ids'][0][i])} <> [{patch_inputs['attention_mask'][0][i]}]{mt.tokenizer.decode(patch_inputs['input_ids'][0][i])}"
        )
    logger.debug(f"{'='*50}")

    query_start = find_token_range(
        string=clean_icq[0],
        substring="#",
        tokenizer=mt.tokenizer,
        offset_mapping=clean_offsets,
        occurrence=-1,
    )[-1]
    logger.debug(f"{query_start=}")

    return dict(
        clean_inputs=clean_inputs,
        patch_inputs=patch_inputs,
        query_start=query_start,
        subj_1_range=subj_1_range,
        subj_2_range=subj_2_range,
    )


@dataclass(frozen=True)
class AttributionPatchingResult(DataClassJsonMixin):
    clean_icq: str
    clean_toks: list[str]
    clean_ans: PredictedToken

    patch_icq: str  # h patched from. The fact we care about
    patch_toks: list[str]
    patch_ans: PredictedToken
    corrupt_patch_ans: tuple[int, PredictedToken]

    query_start: int
    subj_1_range: tuple[int, int]
    subj_2_range: tuple[int, int]
    indirect_effects: dict[str, float]


def get_attribution_patching_results_for_icq_pair(
    mt: ModelandTokenizer,
    clean_icq: str,
    clean_entity_pair: tuple[str, str],
    patch_icq: str,
    patch_entity_pair: tuple[str, str],
):
    inps = process_inputs(
        mt=mt,
        clean_icq=clean_icq,
        clean_entity_pair=clean_entity_pair,
        patch_icq=patch_icq,
        patch_entity_pair=patch_entity_pair,
    )
    clean_inputs = inps["clean_inputs"]
    patch_inputs = inps["patch_inputs"]
    query_start = inps["query_start"]
    subj_1_range = inps["subj_1_range"]
    subj_2_range = inps["subj_2_range"]

    patch_ans = predict_next_token(
        mt=mt,
        inputs=patch_inputs,
    )[
        0
    ][0]
    logger.debug(f"{patch_ans=}")

    clean_ans, corrupt_rank = predict_next_token(
        mt=mt,
        inputs=clean_inputs,
        token_of_interest=[(mt.tokenizer.decode(patch_ans.token_id))],
    )
    clean_ans = clean_ans[0][0]
    corrupt_rank = corrupt_rank[0]
    logger.debug(f"{clean_ans=}")
    logger.debug(f"{corrupt_rank=}")

    assert clean_inputs["input_ids"].shape[1] == patch_inputs["input_ids"].shape[1]

    emb_clean = get_hs(
        mt=mt,
        input=clean_inputs,
        locations=[
            (mt.embedder_name, tok_idx)
            for tok_idx in list(range(*subj_1_range)) + list(range(*subj_2_range))
        ],
    )
    emb_patch = get_hs(
        mt=mt,
        input=patch_inputs,
        locations=[
            (mt.embedder_name, tok_idx)
            for tok_idx in list(range(*subj_1_range)) + list(range(*subj_2_range))
        ],
    )
    patch_spec = [
        PatchSpec(
            location=location,
            patch=emb_patch[location],
            clean=emb_clean[location],
        )
        for location in emb_clean.keys()
    ]

    locations = []
    for l_idx in range(mt.n_layer):
        for tok_idx in range(query_start, clean_inputs.input_ids.shape[1]):
            locations.append((mt.layer_name_format.format(l_idx), tok_idx))
            locations.append((mt.mlp_module_name_format.format(l_idx), tok_idx))
            locations.append((mt.attn_module_name_format.format(l_idx), tok_idx))

            for h_idx in range(mt.model.config.num_attention_heads):
                locations.append(
                    (mt.attn_module_name_format.format(l_idx) + f".{h_idx}", tok_idx)
                )

    results = attribution_patching(
        mt=mt,
        clean_inputs=clean_inputs,
        patches=patch_spec,
        interested_locations=locations,
        ans_token_idx=patch_ans.token_id,
    )
    results_processed = {
        f"{module_name}_<>_{tok_idx}": val
        for (module_name, tok_idx), val in results.items()
    }

    return AttributionPatchingResult(
        clean_icq=clean_icq,
        clean_toks=[mt.tokenizer.decode(t) for t in clean_inputs["input_ids"][0]],
        clean_ans=clean_ans,
        patch_icq=patch_icq,
        patch_toks=[mt.tokenizer.decode(t) for t in patch_inputs["input_ids"][0]],
        patch_ans=patch_ans,
        corrupt_patch_ans=corrupt_rank,
        query_start=query_start,
        subj_1_range=subj_1_range,
        subj_2_range=subj_2_range,
        indirect_effects=results_processed,
    )


@dataclass(frozen=False)
class ExperimentResults(DataClassJsonMixin):
    model_name: str
    relation_name: str
    attribution_patching_results: list[AttributionPatchingResult]


def cache_attribution_patching_results(
    model_name: str,
    relation: str,
    known_data_file: str,
    save_dir: Optional[str] = None,
    limit: Optional[int] = None,
):
    mt = ModelandTokenizer(
        model_key=model_name,
        torch_dtype=torch.float16,
    )
    if save_dir is not None:
        save_dir = os.path.join(
            env_utils.DEFAULT_RESULTS_DIR,
            save_dir,
            mt.name.split("/")[-1],
            relation,
        )
        os.makedirs(save_dir, exist_ok=True)

    cached_known_dir = os.path.join(
        env_utils.DEFAULT_DATA_DIR, "bridge_cached", mt.name.split("/")[-1]
    )
    with open(os.path.join(cached_known_dir, known_data_file), "r") as f:
        json_data = json.load(f)
    relation_icq = None
    for rel in json_data["relations"]:
        if rel["name"] == relation:
            relation_icq = BridgeRelation.from_dict(rel)
            break
    assert (
        relation_icq is not None
    ), f"{relation=} is not found. Available relations: {[r['name'] for r in json_data['relations']]}"

    dataset = BridgeDataset(relations=[relation_icq])
    experiment_results = ExperimentResults(
        model_name=mt.name,
        relation_name=relation_icq.name,
        attribution_patching_results=[],
    )
    limit = len(dataset) if limit is None else limit
    for idx in range(limit):
        logger.info(f"Processing {idx+1}/{limit}")
        patch_icq = dataset[idx]
        clean_idx = sample_icq_with_different_bridge(
            dataset.examples, dataset.examples[idx]
        )
        clean_icq = dataset[clean_idx]

        logger.debug(f"clean_icq: {dataset.examples[clean_idx]}")
        logger.debug(f"patch_icq: {dataset.examples[idx]}")

        atp_result = get_attribution_patching_results_for_icq_pair(
            mt=mt,
            clean_icq=clean_icq,
            clean_entity_pair=dataset.examples[clean_idx].entity_pair,
            patch_icq=patch_icq,
            patch_entity_pair=dataset.examples[idx].entity_pair,
        )

        experiment_results.attribution_patching_results.append(atp_result)

        if save_dir is not None:
            with open(os.path.join(save_dir, f"{relation_icq.name}.json"), "w") as f:
                f.write(experiment_results.to_json(indent=2))
    return experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        choices=["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"],
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--relation",
        type=str,
        choices=[
            "architect_building",
            "movie_actor",
            "sport_players",
            "superpower_characters",
        ],
        default="sport_players",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="attribution_patching",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--known_data",
        type=str,
        default="filtered_2024-07-30T17:30:08.336365.json",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    logger.info(f"{args=}")

    kwargs = dict(
        model_name=args.model,
        relation=args.relation,
        known_data_file=args.known_data,
        save_dir=args.save_dir,
        limit=args.limit if args.limit > 0 else None,
    )

    cache_attribution_patching_results(**kwargs)
