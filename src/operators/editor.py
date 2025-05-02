import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.functional import (
    free_gpu_cache,
    get_hs,
    get_module_nnsight,
    interpret_logits,
    prepare_input,
)
from src.models import ModelandTokenizer
from src.operators.utils import (
    Order1Approx,
    get_inputs_and_intervention_range,
    get_lm_head_row,
    patch,
)
from src.utils.typing import PredictedToken

logger = logging.getLogger(__name__)


def get_edit_delta(
    mt: ModelandTokenizer,
    orig: int | torch.Tensor,
    targ: int | torch.Tensor,
    W_inv: torch.Tensor,
    V_directions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    def token_to_logit(token: int) -> torch.Tensor:
        decoder_row = get_lm_head_row(mt, token)
        return patch(
            h=decoder_row,
            mt=mt,
            inp_layer=mt.layer_names[-1],
            out_layer=mt.lm_head_name,
        )

    orig = token_to_logit(orig) if isinstance(orig, int) else orig
    targ = token_to_logit(targ) if isinstance(targ, int) else targ

    assert orig.shape == targ.shape, f"{orig.shape} != {targ.shape}"
    assert W_inv.shape[1] == orig.shape[0]

    logit_delta = targ - orig
    if V_directions is not None:
        V_directions = V_directions.to(mt.dtype).to(mt.device)
        logit_delta = V_directions.T @ (V_directions @ logit_delta)

    repr_delta = W_inv.to(mt.dtype).to(mt.device) @ logit_delta

    return repr_delta


@dataclass(frozen=False, kw_only=True)
class EditResults:
    original_predictions: list[PredictedToken]
    edited_predictions: list[PredictedToken]

    original_logits: torch.Tensor
    edited_logits: torch.Tensor

    original_generations: Optional[list[str]] = field(default=None)
    edited_generations: Optional[list[str]] = field(default=None)


def apply_jacobian_inv_edit(
    mt: ModelandTokenizer,
    prompt: str,
    subject: str,
    target_tok: int,
    o1_approxes: dict[str, Order1Approx],
    layers: list[str] = [],
    edit_rank: int = 1200,
    num_generations: int = 5,
    num_tokens: int = 20,
    V_directions: Optional[torch.Tensor] = None,
) -> EditResults:
    free_gpu_cache()

    inputs, subj_range = get_inputs_and_intervention_range(mt, prompt, subject)
    subj_ends = subj_range[1] - 1

    with mt.trace(inputs) as tr:
        orig_logits = mt.output.logits.save()

    orig_output = interpret_logits(logits=orig_logits[0, -1], tokenizer=mt.tokenizer)

    if len(layers) == 0:
        layers = list(o1_approxes.keys())

    logger.info(f"{edit_rank=}")
    W_invs = {l: o1_approxes[l].jacobian_inv(rank=edit_rank) for l in layers}

    #! nnsight quirk: can't access h.norm() directly inside trace. need 2 forward passes.
    #! anyway to do this with a single forward pass?
    hs_subj = get_hs(
        mt=mt,
        input=inputs,
        locations=[(layer_name, subj_ends) for layer_name in layers],
        return_dict=True,
    )

    # for l in layers:
    #     corner = CornerOperator(
    #         corner=o1_approxes[l].calculated_at,
    #         mt=mt,
    #         layer=l,
    #         class_indices=[target_tok.item(), orig_output[0].token_id],
    #     )
    #     out = corner(hs_subj[(l, subj_ends)], return_logits=True)
    #     logger.info(f"{l} => {out.top_predictions}")

    repr_diffs = {
        l: get_edit_delta(
            mt=mt,
            targ=target_tok.item(),
            orig=orig_output[0].token_id,
            # orig=patch(
            #     h=o1_approxes[l].calculated_at,
            #     mt=mt,
            #     inp_layer=l,
            # ),
            # orig=CornerOperator(
            #     corner=o1_approxes[l].calculated_at,
            #     mt=mt,
            #     layer=l,
            #     class_indices=[target_tok.item(), orig_output[0].token_id],
            # )(hs_subj[(l, subj_ends)], return_logits=True).logits,
            W_inv=W_invs[l],
            V_directions=V_directions,
        )
        for l in layers
    }

    normalized_deltas = {}
    for layer_name in layers:
        repr_diff = repr_diffs[layer_name].to(mt.device)
        cur_h = hs_subj[(layer_name, subj_ends)].to(mt.device)
        normalized_deltas[layer_name] = (
            3 * (repr_diff * cur_h.norm() / repr_diff.norm()) / 2
        )

        # normalized_deltas[layer_name] = repr_diff

    with mt.trace(inputs) as tr:
        for layer_name in layers:
            repr_diff = repr_diffs[layer_name]
            layer = get_module_nnsight(mt, layer_name)

            # TODO(arnab): normalize the delta as repr_diff * norm(hs) / norm(repr_diff)
            # layer.output[0][:, subj_range, :] += repr_diff

            layer.output[0][:, subj_ends, :] += normalized_deltas[layer_name]

        edited_logits = mt.output.logits.save()

    edited_output = interpret_logits(
        logits=edited_logits[0, -1], tokenizer=mt.tokenizer
    )

    free_gpu_cache()

    batch_inputs = prepare_input(
        prompts=[prompt],
        tokenizer=mt,
        device=mt.device,
        n_gen_per_prompt=num_generations,
    )

    orig_batch_out = mt._model.generate(
        **batch_inputs,
        max_new_tokens=num_tokens,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True,
    )
    orig_generations = mt.tokenizer.batch_decode(
        orig_batch_out.sequences, skip_special_tokens=True
    )

    with mt.generate(
        batch_inputs,
        max_new_tokens=num_tokens,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True,
    ) as gen_trace:
        for layer_name in layers:
            layer = get_module_nnsight(mt, layer_name)
            # layer.output[0][:, subj_range, :] += repr_diff
            layer.output[0][:, subj_ends, :] += normalized_deltas[layer_name]
        edited_batch_out = mt.generator.output.save()

    edited_generations = mt.tokenizer.batch_decode(
        edited_batch_out.sequences, skip_special_tokens=True
    )

    return EditResults(
        original_predictions=orig_output,
        edited_predictions=edited_output,
        original_logits=orig_logits,
        edited_logits=edited_logits,
        original_generations=orig_generations,
        edited_generations=edited_generations,
    )
