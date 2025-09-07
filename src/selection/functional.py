import logging
from itertools import product

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
    repeat_kv,
)
from src.models import ModelandTokenizer
from src.tokens import find_token_range, insert_padding_before_pos, prepare_input
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)


def get_patches_to_verify_independent_enrichment(
    prompt: str,
    options: list[str],
    pivot: str,
    mt: ModelandTokenizer,
    bare_prompt_template: str = "Option: {}",
    tokenized_prompt: TokenizerOutput | None = None,
):
    if not tokenized_prompt or (
        tokenized_prompt and "offset_mapping" not in tokenized_prompt
    ):
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
            f'opt="{mt.tokenizer.decode(tokenized_prompt.input_ids[0][range(*opt_range)])}" | bare_opt="{mt.tokenizer.decode(bare_tokenized.input_ids[0][range(*bare_opt_range)])}"'
        )
        assert (
            opt_range[1] - opt_range[0] == bare_opt_range[1] - bare_opt_range[0]
        ), f"Option range {opt_range} and bare option range {bare_opt_range} do not match for option '{opt}' in prompt '{prompt}'"

        range_diff = opt_range[0] - bare_opt_range[0]
        bare_tokenized = insert_padding_before_pos(
            inp=bare_tokenized,
            token_position=0,
            pad_len=range_diff,
            pad_id=mt.tokenizer.bos_token_id,
            fill_attn_mask=False,
        )
        bare_opt_range = opt_range
        logger.debug(
            f'After adjusted {bare_opt_range=}: bare_opt="{mt.tokenizer.decode(bare_tokenized.input_ids[0][range(*bare_opt_range)])}"'
        )

        bare_hs = get_hs(
            mt=mt,
            input=bare_tokenized,
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
    prompt: str,
    mt: ModelandTokenizer,
    heads: list[tuple[int, int]],
    tokenized_prompt: TokenizerOutput | None = None,
    visualize_individual_heads: bool = False,
    layers: list[int] | None = None,
    value_weighted: bool = False,
    generate_full_answer: bool = False,
    ablate_possible_ans_info_from_options: bool = False,
    options: list[str] | None = None,
    pivot: str | None = None,
    bare_prompt_template=" Options: {}",
    query_index: int = -1,
    query_patches: list[PatchSpec] = [],
    start_from: int = 1,
):
    if tokenized_prompt is None:
        tokenized_prompt = prepare_input(
            tokenizer=mt,
            prompts=prompt,
            return_offsets_mapping=True,
        )
    if ablate_possible_ans_info_from_options:
        assert (
            options is not None and pivot is not None
        ), "Options and pivot must be provided if ablate_possible_ans_info_from_options is True"
        patches = get_patches_to_verify_independent_enrichment(
            prompt=prompt,
            options=options,
            pivot=pivot,
            mt=mt,
            tokenized_prompt=tokenized_prompt,
            bare_prompt_template=bare_prompt_template,
        )
    else:
        patches = []
    patches = patches + query_patches

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
                    start_from=start_from,
                )

        logger.info("Combined attention matrix for all heads")
        combined_matrix = torch.stack(combined).mean(dim=0)
        visualize_attn_matrix(
            attn_matrix=combined_matrix,
            tokens=attn_matrices.tokenized_prompt,
            q_index=query_index,
            start_from=start_from,
        )

    return ret_dict


@torch.inference_mode()
def cache_q_projections(
    mt: ModelandTokenizer,
    input: TokenizerOutput,
    heads: list[tuple[int, int]],  # (layer_idx, head_idx)
    token_indices: list[list[int]],
    return_output: bool = False,
    projection_signature: str = ".q_proj",
):
    batch_size = input.input_ids.shape[0]
    assert len(token_indices) == batch_size, f"{len(token_indices)=} != {batch_size=}"
    layer_to_head = {}
    for layer_idx, head_idx in heads:
        if layer_idx not in layer_to_head:
            layer_to_head[layer_idx] = []
        layer_to_head[layer_idx].append(head_idx)

    seq_len = input.input_ids.shape[1]
    n_heads = mt.config.num_attention_heads
    # head_dim = mt.n_embd // n_heads
    head_dim = get_module_nnsight(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim
    group_size = n_heads // mt.config.num_key_value_heads
    q_module_projections_per_layer = {}
    with mt.trace(input) as tracer:  # noqa
        for layer_idx, head_indices in layer_to_head.items():
            q_proj_name = (
                mt.attn_module_name_format.format(layer_idx) + projection_signature
            )
            q_proj_module = get_module_nnsight(mt, q_proj_name)
            q_module_projections_per_layer[q_proj_name] = q_proj_module.output.save()

        if return_output:
            output = mt.output.save()

    q_projections = [{} for _ in range(batch_size)]
    for layer_idx, head_indices in layer_to_head.items():
        q_proj_name = (
            mt.attn_module_name_format.format(layer_idx) + projection_signature
        )
        # print(q_proj_name)
        q_proj_out = (
            q_module_projections_per_layer[q_proj_name]
            .view(batch_size, seq_len, -1, head_dim)
            .transpose(1, 2)
        )
        if projection_signature in [".k_proj", ".v_proj"] and group_size != 1:
            q_proj_out = repeat_kv(q_proj_out, n_rep=group_size)
        # print(q_proj_out.shape, q_proj_out.norm())
        for prompt_idx in range(batch_size):
            for head_idx in head_indices:
                for token_idx in token_indices[prompt_idx]:
                    q_projections[prompt_idx][(layer_idx, head_idx, token_idx)] = (
                        q_proj_out[prompt_idx, head_idx, token_idx]
                    )

    if return_output:
        return q_projections, output
    return q_projections
