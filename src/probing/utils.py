import copy
import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from dataclasses_json import DataClassJsonMixin

from src.functional import ASK_ORACLE_MODEL
from src.models import ModelandTokenizer, is_llama_variant
from src.tokens import find_token_range, prepare_input
from src.utils.typing import ArrayLike
import os
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class ProbingPrompt(DataClassJsonMixin):
    prompt: str
    entities: tuple[str, str]

    model_key: str
    tokenized: dict[str, torch.Tensor]

    entity_ranges: tuple[tuple[int, int], tuple[int, int]]
    query_range: tuple[int, int]


def prepare_probing_input(
    mt: ModelandTokenizer,
    entities: tuple[str, str],
    prefix: str = "Find a common link or relation between the 2 entities",
    answer_marker: str = "\nA:",
    question_marker: str = "\nQ:",
    block_separator: str = "\n#",
    is_a_reasoning_model: bool = False,
    answer_prefix: str = "",
    return_offsets_mapping: bool = False,
) -> ProbingPrompt:

    prompt = f"""{prefix.strip()}{block_separator}{question_marker}{entities[0]} and {entities[1]}{answer_marker}{answer_prefix}"""
    if is_a_reasoning_model:
        thinking_instructions = "Try to keep your thinking is less than 5 sentences. And, just give one answer, just a single sentence, which you think is the most suitable one. Put your answer within \\boxed{}."
        prompt = f"{prompt}\n{thinking_instructions}\n<think>"

    tokenized = prepare_input(prompts=prompt, tokenizer=mt, return_offsets_mapping=True)
    offset_mapping = tokenized["offset_mapping"][0]

    entity_ranges = tuple(
        [
            find_token_range(
                string=prompt,
                substring=entity,
                tokenizer=mt,
                offset_mapping=offset_mapping,
                occurrence=-1,
            )
            for entity in entities
        ]
    )
    query_len = prepare_input(
        prompts=answer_marker, tokenizer=mt, add_special_tokens=False
    )["input_ids"].shape[1]
    query_range = (
        tokenized.input_ids.shape[1] - query_len,
        tokenized.input_ids.shape[1],
    )

    tokenized = dict(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
    )
    if return_offsets_mapping:
        tokenized["offset_mapping"] = [offset_mapping]

    return ProbingPrompt(
        prompt=prompt,
        entities=entities,
        model_key=mt.name.split("/")[-1],
        tokenized=tokenized,
        entity_ranges=entity_ranges,
        query_range=query_range,
    )


def get_lm_generated_answer(
    mt: ModelandTokenizer,
    prompt: ProbingPrompt,
    block_separator: str = "\n#",
    is_a_reasoning_model: bool = False,
):
    with mt.generate(
        prompt.tokenized,
        # dict(
        #     input_ids=prompt.tokenized.input_ids,
        #     attention_mask=prompt.tokenized.attention_mask,
        # ),
        max_new_tokens=50 if is_a_reasoning_model == False else 1000,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    ) as gen_trace:
        output = mt.generator.output.save()

    generation = mt.tokenizer.decode(
        output.sequences[0][prompt.tokenized["input_ids"].shape[-1] :],
        skip_special_tokens=False,
    ).strip()

    # print(generation)

    if is_a_reasoning_model == False:
        if block_separator in generation:
            generation = generation.split(block_separator)[0].strip()
    else:
        if "\\boxed{" in generation:
            monologue = generation.split("<think>")[-1].split("</think>")[0].strip()
            logger.debug(f"{monologue=}")
            generation = generation.split("\\boxed{")[1].split("}")[0].strip()
            if "{" in generation:
                generation = generation.split("{")[1].split("}")[0].strip()

    return generation


def check_if_answer_is_correct(
    answer: str,
    keywords: list[str] = [],
    oracle_model: Optional[Literal["gpt4o", "claude"]] = "claude",
    entities: tuple[str, str] = None,
) -> bool:
    """
    will return true if any keyword is found in the answer.
    if not and the oracle is not None, will ask the oracle model for verification.
    """

    #! use this only if the answer is not None
    assert (
        answer.startswith("None") == False
    ), f'Pass a valid answer to check, passed: "{answer}"'

    if any([keyword in answer for keyword in keywords]):
        return True

    elif oracle_model is not None:
        question = f"""Do you think the following answer is a good connection or relation between the entities {entities[0]} and {entities[1]}?
Your answer should start with "Yes" or "No". If the answer is "No", please provide your reasoning. Otherwise, just say "Yes".
\n\n{answer}"""
        oracle_response = ASK_ORACLE_MODEL[oracle_model](prompt=question)
        logger.info(f"{oracle_response=}")
        return oracle_response.startswith("Yes")

    return False


def map_latent_keys_to_str(dct):
    cpy = {}
    for key, value in dct.items():
        cpy["_<>_".join([str(k) for k in key])] = value
    return cpy


@dataclass
class ProbingLatents(DataClassJsonMixin):
    prompt: ProbingPrompt
    latents: dict[tuple[str, int] | str, ArrayLike]
    lm_answer: Optional[str] = None

    def get_dict(self):
        dct = map_latent_keys_to_str(self.latents)
        dct.update(self.prompt.to_dict())
        return dct

    @staticmethod
    def from_npz(npz_file):
        if isinstance(npz_file, str):
            npz_file = np.load(npz_file, allow_pickle=True)
        prompt = ProbingPrompt(
            prompt=npz_file["prompt"].item(),
            entities=npz_file["entities"].tolist(),
            model_key=npz_file["model_key"].item(),
            tokenized=npz_file["tokenized"].item(),
            entity_ranges=npz_file["entity_ranges"].tolist(),
            query_range=npz_file["query_range"].tolist(),
        )

        if "lm_answer" in npz_file.files:
            lm_answer = npz_file["lm_answer"].item()
        else:
            lm_answer = None

        avoid = [
            "prompt",
            "entities",
            "model_key",
            "tokenized",
            "entity_ranges",
            "query_range",
            "lm_answer",
        ]
        latents = {}
        for key in npz_file.files:
            if key not in avoid and key != "allow_pickle":
                location = key.split("_<>_")
                location[1] = int(location[1])
                latents[tuple(location)] = torch.Tensor(npz_file[key])

        return ProbingLatents(
            prompt=prompt,
            latents=latents,
            lm_answer=lm_answer,
        )


def load_probing_activations(
    token_query_pos: int,
    latent_root: str,
    layers: list[str],
    limit: int = None,
):
    classes = os.listdir(latent_root)
    activations = {cls: [] for cls in classes}
    for cls in classes:
        cls_dir = os.path.join(latent_root, cls)
        npz_files = os.listdir(cls_dir)
        if limit is not None:
            npz_files = npz_files[:limit]
        logger.info(f"{cls=} ... loading {len(npz_files)} latents ...")

        for npz in tqdm(npz_files, mininterval=5):
            npz_path = os.path.join(cls_dir, npz)
            latents_npz = np.load(npz_path, allow_pickle=True)
            cached_latents = ProbingLatents.from_npz(latents_npz)
            cur_activations = {}
            for layer in layers:
                location = (
                    layer,
                    list(range(*cached_latents.prompt.query_range))[token_query_pos],
                )
                cur_activations[layer] = cached_latents.latents[location]
            activations[cls].append(cur_activations)
            if limit is not None and len(activations[cls]) >= limit:
                break

    return activations
