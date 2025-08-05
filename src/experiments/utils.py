import logging
from itertools import product

import torch

from src.attention import (
    get_attention_matrices,
    visualize_attn_matrix,
    visualize_average_attn_matrix,
)
from src.functional import PatchSpec, generate_with_patch, get_hs, interpret_logits
from src.models import ModelandTokenizer
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
        logger.debug(
            f'{opt} | {opt_range=} | {bare_opt_range=} | "{mt.tokenizer.decode(tokenized_prompt.input_ids[0][range(*opt_range)])}"'
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
    prompt: str,
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
):
    tokenized_prompt = prepare_input(
        tokenizer=mt,
        prompts=prompt,
        return_offsets_mapping=True,
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
                    q_index=-1,
                    start_from=1,
                )

        logger.info("Combined attention matrix for all heads")
        combined_matrix = torch.stack(combined).mean(dim=0)
        visualize_attn_matrix(
            attn_matrix=combined_matrix,
            tokens=attn_matrices.tokenized_prompt,
            q_index=-1,
            start_from=1,
        )

    return ret_dict
