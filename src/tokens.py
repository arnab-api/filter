import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, overload

import baukit
import torch
import transformers
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import ModelandTokenizer, determine_device, unwrap_tokenizer
from src.utils.env_utils import DEFAULT_MODELS_DIR
from src.utils.typing import Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


def maybe_prefix_bos(tokenizer, prompt: str) -> str:
    """Prefix prompt with EOS token if model has no special start token."""
    tokenizer = unwrap_tokenizer(tokenizer)
    if hasattr(tokenizer, "bos_token"):
        prefix = tokenizer.bos_token
        if not prompt.startswith(prefix):
            prompt = prefix + " " + prompt
    return prompt


def prepare_offset_mapping(string, tokenized, special_tokens):
    """LLaMA3 tokenizer in Huggingface is buggy. This function is a workaround for the bug."""
    """
    <Test>
    
    prompts = ["The Eiffle Tower is located in", "The Space Needle is located in"]
    inp = prepare_input(
        prompts = prompts,
        tokenizer=mt,
        return_offsets_mapping=True,
        device="cuda"
    )

    i=1 # <to be changed>
    for token_id, offset in zip(inp["input_ids"][i], inp["offset_mapping"][i]):
        print(f"`{tokenizer.decode(token_id)}`, {offset=} | `{prompts[i][offset[0]:offset[1]]}`")

    """
    # logger.debug(f"{special_tokens}")
    offset_mapping = []
    end = 0
    for token in tokenized:
        if token in special_tokens:
            offset_mapping.append((end, end))
            continue
        # print(f"{string[end:].find(token)} | {end=}, {token=}, {string[end:]}")
        next_tok_idx = string[end:].find(token)
        assert next_tok_idx != -1, f"{token} not found in {string[end:]}"
        assert next_tok_idx in [
            0,
            1,
        ], f"{token} not found at the beginning of the string"

        start = end
        end = start + string[end:].find(token) + len(token)
        offset_mapping.append((start, end))
    return offset_mapping


def prepare_input(
    prompts: str | list[str],
    tokenizer: ModelandTokenizer | Tokenizer,
    n_gen_per_prompt: int = 1,
    device: torch.device = "cpu",
    add_bos_token: bool = False,
    return_offsets_mapping=False,
) -> TokenizerOutput:
    """Prepare input for the model."""
    if isinstance(tokenizer, ModelandTokenizer):
        device = determine_device(
            tokenizer
        )  # if tokenizer type is ModelandTokenizer, get device and ignore the passed device
    calculate_offsets = return_offsets_mapping and (
        isinstance(tokenizer, ModelandTokenizer) and "llama-3" in tokenizer.name.lower()
    )

    tokenizer = unwrap_tokenizer(tokenizer)
    prompts = [prompts] if isinstance(prompts, str) else prompts
    if add_bos_token:
        prompts = [maybe_prefix_bos(tokenizer, p) for p in prompts]
    prompts = [p for p in prompts for _ in range(n_gen_per_prompt)]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        return_offsets_mapping=return_offsets_mapping,
    )

    if calculate_offsets:
        offsets = []
        for i in range(len(prompts)):
            tokenized = [tokenizer.decode(t) for t in inputs["input_ids"][i]]
            offsets.append(
                prepare_offset_mapping(
                    string=prompts[i],
                    tokenized=tokenized,
                    special_tokens=tokenizer.all_special_tokens,
                )
            )
        inputs["offset_mapping"] = torch.tensor(offsets)

    inputs = inputs.to(device)
    return inputs


def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[Tokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')

    # logger.debug(f"Found substring in string {string.count(substring)} times")

    if occurrence < 0:
        # If occurrence is negative, count from the right.
        char_start = string.rindex(substring)
        for _ in range(-1 - occurrence):
            try:
                char_start = string.rindex(substring, 0, char_start)
            except ValueError as error:
                raise ValueError(
                    f"could not find {-occurrence} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    else:
        char_start = string.index(substring)
        for _ in range(occurrence):
            try:
                char_start = string.index(substring, char_start + 1)
            except ValueError as error:
                raise ValueError(
                    f"could not find {occurrence + 1} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    char_end = char_start + len(substring)

    # logger.debug(
    #     f"char range: [{char_start}, {char_end}] => `{string[char_start:char_end]}`"
    # )

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = prepare_input(
            string, return_offsets_mapping=True, tokenizer=tokenizer, **kwargs
        )
        offset_mapping = tokens.offset_mapping

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        # logger.debug(f"{index=} | token range: [{token_char_start}, {token_char_end}]")
        if token_char_start == token_char_end:
            # Skip special tokens # ! Is this the proper way of doing this?
            continue
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


def insert_padding_before_subj(
    inp: TokenizerOutput,
    subj_range: tuple[int, int],
    subj_ends: int,
    pad_id: int,
    fill_attn_mask: bool = False,
):
    """

    Inserts padding tokens before the subject in the query to balance the input tensor.

    TEST:

    for idx, (tok_id, attn_mask) in enumerate(zip(clean_inputs.input_ids[0], clean_inputs.attention_mask[0])):
        print(f"{idx=} [{attn_mask}] | {mt.tokenizer.decode(tok_id)}")

    """
    pad_len = subj_ends - subj_range[1]
    inp["input_ids"] = torch.cat(
        [
            inp.input_ids[:, : subj_range[0]],
            torch.full(
                (1, pad_len),
                pad_id,
                dtype=inp.input_ids.dtype,
                device=inp.input_ids.device,
            ),
            inp.input_ids[:, subj_range[0] :],
        ],
        dim=1,
    )

    inp["attention_mask"] = torch.cat(
        [
            inp.attention_mask[:, : subj_range[0]],
            torch.full(
                (1, pad_len),
                fill_attn_mask,
                dtype=inp.attention_mask.dtype,
                device=inp.attention_mask.device,
            ),
            inp.attention_mask[:, subj_range[0] :],
        ],
        dim=1,
    )
    return inp


def align_bridge_entities_in_query(
    mt: ModelandTokenizer,
    clean_ques: str,
    clean_entity_pair: tuple[str, str],
    patch_ques: str,
    patch_entity_pair: tuple[str, str],
) -> tuple[TokenizerOutput, TokenizerOutput, int]:
    clean_inputs = prepare_input(
        prompts=clean_ques, tokenizer=mt, return_offsets_mapping=True
    )

    clean_offsets = clean_inputs.pop("offset_mapping")[0]
    clean_subj_ranges = [
        find_token_range(
            string=clean_ques,
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
        prompts=patch_ques, tokenizer=mt, return_offsets_mapping=True
    )
    patch_offsets = patch_inputs.pop("offset_mapping")[0]
    patch_subj_ranges = [
        find_token_range(
            string=patch_ques,
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
        string=clean_ques,
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
