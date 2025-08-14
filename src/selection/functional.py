import logging
from itertools import product
from typing import Optional

import torch

from src.attention import (
    get_attention_matrices,
    visualize_attn_matrix,
    visualize_average_attn_matrix,
)
from src.functional import (
    PatchSpec,
    generate_with_patch,
    get_hs,
    get_module_nnsight,
    interpret_logits,
    patch_with_baukit,
)
from src.models import ModelandTokenizer
from src.selection.data import SelectionSample
from src.selection.utils import get_first_token_id
from src.tokens import find_token_range, prepare_input
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)


def get_patches_to_verify_independent_enrichment(
    prompt: str,
    options: list[str],
    pivot: str,
    mt: ModelandTokenizer,
    bare_prompt_template: str = " The fact that {}",
    tokenized_prompt: TokenizerOutput | None = None,
):
    if tokenized_prompt and "offset_mapping" not in tokenized_prompt:
        tokenized_prompt = prepare_input(
            tokenizer=mt,
            prompts=prompt,
            return_offsets_mapping=True,
        )

    offsets = tokenized_prompt.pop("offset_mapping")[0]
    patches = []
    for opt in options:
        opt_range = find_token_range(
            tokenizer=mt,
            string=prompt,
            substring=opt,
            offset_mapping=offsets,
        )
        # print(
        #     [
        #         mt.tokenizer.decode(tokenized_prompt.input_ids[0][i])
        #         for i in range(*opt_range)
        #     ]
        # )

        if mt.tokenizer.decode(tokenized_prompt.input_ids[0][opt_range[0]]) == "\n":
            # If the option starts with a newline, we need to adjust the range
            opt_range = (opt_range[0] + 1, opt_range[1])

        bare_prompt = bare_prompt_template.format(opt)
        bare_tokenized = prepare_input(
            tokenizer=mt,
            prompts=bare_prompt,
            return_offsets_mapping=True,
        )
        bare_offsets = bare_tokenized.pop("offset_mapping")[0]
        bare_opt_range = find_token_range(
            tokenizer=mt,
            string=bare_prompt,
            substring=opt,
            offset_mapping=bare_offsets,
        )
        logger.debug(f"{opt} | {opt_range=} | {bare_opt_range=}")
        logger.debug(
            f'"{mt.tokenizer.decode(tokenized_prompt.input_ids[0][range(*opt_range)])}"'
        )
        logger.debug(
            f'"{mt.tokenizer.decode(bare_tokenized.input_ids[0][range(*bare_opt_range)])}"'
        )
        assert (
            opt_range[1] - opt_range[0] == bare_opt_range[1] - bare_opt_range[0]
        ), f"Option range {opt_range} and bare option range {bare_opt_range} do not match for option '{opt}' in prompt '{prompt}'"

        bare_hs = get_hs(
            mt=mt,
            input=bare_prompt,
            locations=list(product(mt.layer_names, range(*bare_opt_range))),
            return_dict=True,
            patches=[],
        )

        for bare_idx, clean_idx in zip(range(*bare_opt_range), range(*opt_range)):
            patches.extend(
                PatchSpec(
                    location=(module_name, clean_idx),
                    patch=bare_hs[(module_name, bare_idx)],
                )
                for module_name in mt.layer_names
            )

    return patches


def verify_head_patterns(
    prompt: str | TokenizerOutput,
    options: list[str],
    pivot: str,
    mt: ModelandTokenizer,
    heads: list[tuple[int, int]],
    visualize_individual_heads: bool = False,
    layers: list[int] | None = None,
    value_weighted: bool = False,
    generate_full_answer: bool = False,
    ablate_possible_ans_info_from_options: bool = False,
    bare_prompt_template=" The fact that {}",
    query_index: int = -1,
    query_patches: list[PatchSpec] = [],
):
    tokenized_prompt = (
        prepare_input(
            tokenizer=mt,
            prompts=prompt,
            return_offsets_mapping=True,
        )
        if isinstance(prompt, str)
        else prompt
    )
    patches = (
        get_patches_to_verify_independent_enrichment(
            prompt=prompt,
            options=options,
            pivot=pivot,
            mt=mt,
            tokenized_prompt=tokenized_prompt,
            bare_prompt_template=bare_prompt_template,
        )
        if ablate_possible_ans_info_from_options
        else []
    )
    patches = patches + query_patches
    print(len(patches), "patches to ablate possible answer information from options")

    ret_dict = {}
    if generate_full_answer:
        gen = generate_with_patch(
            mt=mt,
            inputs=prompt,
            n_gen_per_prompt=1,
            max_new_tokens=30,
            patches=patches,
            remove_prefix=True,
            do_sample=False,
        )[0]
        logger.debug(f'Generated full answer: "{gen}"')
        ret_dict["full_answer"] = gen

    attn_matrices = get_attention_matrices(
        input=tokenized_prompt, mt=mt, value_weighted=value_weighted, patches=patches
    )
    logits = attn_matrices.logits

    predictions = interpret_logits(
        tokenizer=mt.tokenizer,
        logits=logits,
    )
    logger.debug(f"Predictions: {[str(p) for p in predictions]}")
    ret_dict["predictions"] = predictions
    ret_dict["logits"] = logits
    ret_dict["attn_matrices"] = attn_matrices

    if layers is not None and len(layers) > 0:
        visualize_average_attn_matrix(
            mt=mt,
            attn_matrices=attn_matrices,
            prompt=prompt,
            tokenized=tokenized_prompt,
            layer_window=layers,
            q_index=query_index,
        )
        print("=" * 70)

    if heads is not None and len(heads) > 0:
        combined = []
        for layer_idx, head_idx in heads:
            head_matrix = torch.Tensor(
                attn_matrices.attention_matrices[layer_idx, head_idx].squeeze()
            )
            combined.append(head_matrix)
            if visualize_individual_heads:
                logger.info(f"Layer: {layer_idx}, Head: {head_idx}")
                visualize_attn_matrix(
                    attn_matrix=head_matrix,
                    tokens=attn_matrices.tokenized_prompt,
                    q_index=query_index,
                    start_from=1,
                )

        logger.info("Combined attention matrix for all heads")
        combined_matrix = torch.stack(combined).mean(dim=0)
        visualize_attn_matrix(
            attn_matrix=combined_matrix,
            tokens=attn_matrices.tokenized_prompt,
            q_index=query_index,
            start_from=1,
        )

    return ret_dict


@torch.inference_mode()
def cache_q_projections(
    mt: ModelandTokenizer,
    input: TokenizerOutput,
    query_locations: list[tuple[int, int, int]],  # (layer_idx, head_idx, query_idx)
    return_output: bool = False,
):
    layer_to_hq = {}
    for layer_idx, head_idx, query_idx in query_locations:
        if layer_idx not in layer_to_hq:
            layer_to_hq[layer_idx] = []
        layer_to_hq[layer_idx].append((head_idx, query_idx))

    q_projections = {}
    batch_size = input.input_ids.shape[0]
    seq_len = input.input_ids.shape[1]
    n_heads = mt.config.num_attention_heads
    head_dim = mt.n_embd // n_heads
    with mt.trace(input) as tracer:  # noqa
        for layer_idx, query_locs in layer_to_hq.items():
            q_proj_name = mt.attn_module_name_format.format(layer_idx) + ".q_proj"
            q_proj_module = get_module_nnsight(mt, q_proj_name)
            q_proj_out = q_proj_module.output.view(
                batch_size, seq_len, n_heads, head_dim
            ).transpose(1, 2)
            for head_idx, query_idx in query_locs:
                q_projections[(layer_idx, head_idx, query_idx)] = (
                    q_proj_out[:, head_idx, query_idx, :].squeeze().save()
                )

        if return_output:
            output = mt.output.save()

    if return_output:
        return q_projections, output
    return q_projections


@torch.inference_mode()
def validate_q_proj_ie_on_sample_pair(
    mt: ModelandTokenizer,
    clean_sample: SelectionSample,
    patch_sample: SelectionSample,
    heads: list[tuple[int, int]],
    query_indices: int = [-1],
    verify_head_behavior_on: Optional[int] = None,
):
    clean_tokenized = prepare_input(prompts=clean_sample.prompt(), tokenizer=mt)
    patch_tokenized = prepare_input(prompts=patch_sample.prompt(), tokenizer=mt)

    if verify_head_behavior_on is not None:
        logger.info("Verifying head behavior...")

        logger.info(f"Clean Sample >> Ans: {clean_sample.obj}")
        clean_attn_pattern = verify_head_patterns(  # noqa
            prompt=clean_tokenized,
            options=clean_sample.options,
            pivot=clean_sample.subj,
            mt=mt,
            heads=heads,
            generate_full_answer=True,
            query_index=verify_head_behavior_on,
        )

        logger.info(f"Patch Sample >> Ans: {patch_sample.obj}")
        patch_attn_pattern = verify_head_patterns(  # noqa
            prompt=patch_tokenized,
            options=patch_sample.options,
            pivot=patch_sample.subj,
            mt=mt,
            heads=heads,
            generate_full_answer=True,
            query_index=verify_head_behavior_on,
        )

    logger.info(f"Caching the query states for the {len(heads)} heads")

    query_locations = [
        (layer_idx, head_idx, query_idx)
        for layer_idx, head_idx in heads
        for query_idx in query_indices
    ]

    cached_q_states, patch_output = cache_q_projections(
        mt=mt,
        input=patch_tokenized,
        query_locations=query_locations,
        return_output=True,
    )
    q_proj_patches = []
    for (layer_idx, head_idx, query_idx), q_proj in cached_q_states.items():
        q_proj_patches.append(
            PatchSpec(
                location=(
                    mt.attn_module_name_format.format(layer_idx) + ".q_proj",
                    head_idx,
                    query_idx,
                ),
                patch=q_proj,
            )
        )

    patch_logits = patch_output.logits[:, -1, :].squeeze()
    patch_predictions = interpret_logits(
        tokenizer=mt,
        logits=patch_logits,
    )
    logger.info(f"patch_prediction={[str(pred) for pred in patch_predictions]}")

    # interested_tokens = [
    #     patch_sample.ans_token_id,
    #     clean_sample.ans_token_id,
    #     clean_sample.metadata["track_type_obj_token_id"],
    # ]
    interested_tokens = clean_sample.options
    interested_tokens = [
        get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
        for opt in interested_tokens
    ]
    interested_tokens += [patch_sample.ans_token_id]
    # interested_tokens = list(set(interested_tokens))  # remove duplicates #! don't need to, made sure during sampling

    logger.info("clean run")
    clean_output = patch_with_baukit(
        mt=mt,
        inputs=clean_tokenized,
        patches=[],
    )
    clean_logits = clean_output.logits[:, -1, :].squeeze()
    clean_predictions, clean_track = interpret_logits(
        tokenizer=mt,
        logits=clean_logits,
        interested_tokens=interested_tokens,
    )
    logger.info(f"clean_prediction={[str(pred) for pred in clean_predictions]}")
    logger.info(f"clean_track={clean_track}")

    logger.info("patching the q_proj states")
    if verify_head_behavior_on is not None:
        int_attn_pattern = verify_head_patterns(
            prompt=clean_tokenized,
            options=clean_sample.options,
            pivot=clean_sample.subj,
            mt=mt,
            heads=heads,
            query_patches=q_proj_patches,
            generate_full_answer=False,
            query_index=verify_head_behavior_on,
        )
        int_logits = int_attn_pattern["logits"].squeeze()

    else:
        int_out = patch_with_baukit(
            mt=mt,
            inputs=clean_tokenized,
            patches=q_proj_patches,
        )
        int_logits = int_out.logits[:, -1, :].squeeze()

    int_predictions, int_track = interpret_logits(
        tokenizer=mt,
        logits=int_logits,
        interested_tokens=interested_tokens,
    )
    logger.info(f"int_prediction={[str(pred) for pred in int_predictions]}")
    logger.info(f"int_track={int_track}")

    return {
        "clean_sample": clean_sample,
        "patch_sample": patch_sample,
        "clean_predictions": clean_predictions,
        "patch_predictions": patch_predictions,
        "int_predictions": int_predictions,
        "clean_track": clean_track,
        "int_track": int_track,
    }
