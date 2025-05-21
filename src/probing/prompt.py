import logging
import os
from dataclasses import dataclass

import torch
from dataclasses_json import DataClassJsonMixin
import numpy as np

from src.models import ModelandTokenizer
from src.tokens import find_token_range, prepare_input

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
        # thinking_instructions = "Try to keep your thinking is less than 5 sentences. And, just give one answer, just a single sentence, which you think is the most suitable one"
        thinking_instructions = "Just give one answer, in a single line, which you think is the most suitable one"
        prompt = f"{prompt}\n{thinking_instructions}"
        # prompt += "\n<think>"
        messages = [{"role": "user", "content": prompt}]
        prompt = mt.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    # prompt += "<think></think>"

    tokenized = prepare_input(
        prompts=prompt,
        tokenizer=mt,
        return_offsets_mapping=True,  # add_bos_token=True
    )
    offset_mapping = tokenized["offset_mapping"][0]

    positions = [-1, -1] if entities[0] != entities[1] else [-2, -1]
    entity_ranges = tuple(
        [
            find_token_range(
                string=prompt,
                substring=entity,
                tokenizer=mt,
                offset_mapping=offset_mapping,
                occurrence=pos,
            )
            for entity, pos in zip(entities, positions)
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


import numpy as np


class BiAssociationPrefix:
    description = "whether two people share an attribute"
    instruction = """
Given the names of two people, determine whether they share some common link or attribute. Below are the attributes you should consider:"""
    answer_format = """
- nationality => If both are from the same country, respond with `"Yes - they are both <nationality>"`
- profession => If both are in the same profession, respond with `"Yes - they are both <profession>"`.
- alma mater => If both graduated from the same school, respond with `"Yes - they both graduated from <school>"`.
- hobby => If both have the same hobby, respond with `"Yes - they both enjoy <hobby>"`.
- pet => If both have the same pet, respond with `"Yes - they both have a <pet> as their pet"`.
- car => If both have the same car, respond with `"Yes - they both drive a <car>"`.
- allergy => If both have the same allergy, respond with `"Yes - they are both allergic to <allergy>"`.
- favorite food => If both have the same favorite food, respond with `"Yes - they both love <food>"`.
- favorite drink => If both have the same favorite drink, respond with `"Yes - they both love <drink>"`.
- favorite color => If both have the same favorite color, respond with `"Yes - they both love <color>"`.
- biggest fear => If both have the same biggest fear, respond with `"Yes - they are both afraid of <fear>"`.

If you cannot find any connection, just answer "No - <person_1> and <person_2> have nothing in common". Check the following examples for the formatting.
<format>"""

    answer_format_trimmed = """
- nationality => If both are from the same country, respond with `"Yes - they are both <nationality>"`
- profession => If both are in the same profession, respond with `"Yes - they are both <profession>"`.
- alma mater => If both graduated from the same school, respond with `"Yes - they both graduated from <school>"`.

If you cannot find any connection, just answer "No - <person_1> and <person_2> have nothing in common". Check the following examples for the formatting.
<format>"""

    answer_format_2 = """
- nationality => If both are from the same country, respond with `<nationality> - They are both from <country>`.
- profession => If both are in the same profession, respond with `<profession> - They are both <profession>`.
- alma mater => If both graduated from the same school, respond with `<school> - They both graduated from <school>`.
- hobby => If both have the same hobby, respond with `<hobby> - They both enjoy <hobby>`.
- pet => If both have the same pet, respond with `<pet> - They both have a <pet> as their pet`.
- car => If both have the same car, respond with `<car> - They both drive a <car>`.
- allergy => If both have the same allergy, respond with `<allergy> - They are both allergic to <allergy>`.
- favorite food => If both have the same favorite food, respond with `<food> - They both love <food>`.
- favorite drink => If both have the same favorite drink, respond with `<drink> - They both love <drink>`.
- favorite color => If both have the same favorite color, respond with `<color> - They both love <color>`.
- biggest fear => If both have the same biggest fear, respond with `<fear> - They are both afraid of <fear>`.
If you cannot find any connection, just answer "None - <person_1> and <person_2> have nothing in common". Check the following examples for the formatting.
<format>"""

    # instruction = f"{instruction}\n{answer_format}"
    # instruction = f"{instruction}\n{answer_format_2}"
    instruction = f"{instruction}\n{answer_format_trimmed}"

    suffix = "\n</format>\n\n"
    # suffix = ""

    block_separator = "\n#"
    question_marker = "\nQ: "
    answer_marker = "\nA:"

    positive_connections = [
        {
            "entities": ["Person A", "Person B"],
            "connection": "Yes - they are both German.",
            "connection_2": "German - they are both German.",
        },
        {
            "entities": ["Person C", "Person D"],
            "connection": "Yes - they are both doctors.",
            "connection_2": "Doctors - they are both doctors.",
        },
        {
            "entities": ["Person E", "Person F"],
            "connection": "Yes - they both graduated from Harvard University.",
            "connection_2": "Harvard University - they both graduated from Harvard University.",
        },
        # {
        #     "entities": ["Person G", "Person H"],
        #     "connection": "Yes - they both enjoy painting.",
        #     "connection_2": "Painting - they both enjoy painting.",
        # },
        # {
        #     "entities": ["Person I", "Person J"],
        #     "connection": "Yes - they both have a dog as their pet.",
        #     "connection_2": "Dog - they both have a dog as their pet.",
        # },
        # {
        #     "entities": ["Person K", "Person L"],
        #     "connection": "Yes - they both drive a Tesla.",
        #     "connection_2": "Tesla - they both drive a Tesla.",
        # },
        # {
        #     "entities": ["Person M", "Person N"],
        #     "connection": "Yes - they are both allergic to peanuts.",
        #     "connection_2": "Peanuts - they are both allergic to peanuts.",
        # },
        # {
        #     "entities": ["Person O", "Person P"],
        #     "connection": "Yes - they both love sushi.",
        #     "connection_2": "Sushi - they both love sushi.",
        # },
        # {
        #     "entities": ["Person Q", "Person R"],
        #     "connection": "Yes - they both love coffee.",
        #     "connection_2": "Coffee - they both love coffee.",
        # },
        # {
        #     "entities": ["Person S", "Person T"],
        #     "connection": "Yes - they both love blue.",
        #     "connection_2": "Blue - they both love blue.",
        # },
        # {
        #     "entities": ["Person U", "Person V"],
        #     "connection": "Yes - they are both afraid of heights.",
        #     "connection_2": "Heights - they are both afraid of heights.",
        # },
    ]

    negative_connections = [
        {
            "entities": ["Person W", "Person X"],
            "connection": "No - Person W and Person X have nothing in common.",
            "connection_2": "None - Person W and Person X have nothing in common.",
        },
        {
            "entities": ["Person Y", "Person Z"],
            "connection": "No - Person Y and Person Z have nothing in common.",
            "connection_2": "None - Person Y and Person Z have nothing in common.",
        },
    ]

    def __init__(
        self,
        instruction: str = None,
        block_separator: str = None,
        question_marker: str = None,
        answer_marker: str = None,
        positive_connections: list[dict] = None,
        negative_connections: list[dict] = None,
        suffix: str = None,
    ):
        if instruction is not None:
            self.instruction = instruction
        if block_separator is not None:
            self.block_separator = block_separator
        if question_marker is not None:
            self.question_marker = question_marker
        if answer_marker is not None:
            self.answer_marker = answer_marker
        if positive_connections is not None:
            self.positive_connections = positive_connections
        if negative_connections is not None:
            self.negative_connections = negative_connections
        if suffix is not None:
            self.suffix = suffix

    def get_prefix(self, n_valid=4, n_none=2):
        selected_valid = np.random.choice(
            self.positive_connections,
            size=min(n_valid, len(self.positive_connections)),
            replace=False,
        ).tolist()
        selected_none = np.random.choice(
            self.negative_connections,
            size=min(n_none, len(self.negative_connections)),
            replace=False,
        ).tolist()

        connections = selected_valid + selected_none

        np.random.shuffle(connections)
        prefix = self.instruction + "\n"
        if "</format>" in self.suffix and "<format>" not in prefix:
            prefix += "<format>\n"

        for conn in connections:
            prefix += self.block_separator
            prefix += (
                f"{self.question_marker}{conn['entities'][0]} and {conn['entities'][1]}"
            )
            prefix += f"{self.answer_marker} {conn['connection']}"

        return prefix + self.suffix
