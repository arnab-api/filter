import logging
import os
from dataclasses import dataclass

import torch
from dataclasses_json import DataClassJsonMixin
import numpy as np

from src.models import ModelandTokenizer
from src.tokens import find_token_range, prepare_input
from typing import Literal

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
    task_description = """# Task: Find Common Attributes Between Two People
You will be given two people's names. Your job is to determine if they share ANY common attribute from the list below.
"""
    task_description_single = """# Task: Check if Two People Share the Same [attribute]
You will be given two people's names. Your job is to determine if they share the same [attribute]."""

    answer_format_instruction = {
        "format": """
## Response Format:
- If you find a match: "Yes - they [description of what they share]"
- If no match: "No - [Person 1] and [Person 2] have nothing in common"
""",
        "format_2": """
## Response Format:
- If you find a match: "[shared entity] - [description of what they share]"
- If no match: "None - [Person 1] and [Person 2] have nothing in common"
""",
        "format_3": """
## Response Format:
- If you find a match: "Yes - [shared entity] - [description of what they share]"
- If no match: "No - [Person 1] and [Person 2] have nothing in common"
""",
    }

    #     answer_format_instruction_2 = """
    # ## Response Format:
    # - If you find a match: "[shared attribute] - [description of what they share]"
    # - If no match: "None - [Person 1] and [Person 2] have nothing in common"
    # """

    #     answer_format_instruction_3 = """
    # ## Response Format:
    # - If you find a match: "Yes - [shared attribute] - [description of what they share]"
    # - If no match: "No - [Person 1] and [Person 2] have nothing in common"
    # """

    attribute_format = {
        "nationality": {
            "format": "Yes - they are both [nationality]",
            "format_2": "[nationality] - they are both [nationality]",
            "format_3": "Yes - [nationality] - they are both [nationality]",
            "example": {
                "entities": ["Person A", "Person B"],
                "connection": "Yes - they are both German.",
                "connection_2": "German - they are both German.",
                "connection_3": "Yes - German - they are both German.",
            },
            "negative": {
                "format": "No - [person_1] is a [nationality_1] while [person_2] is a [nationality_2]",
                "example": {
                    "entities": ["Person A", "Person B"],
                    "connection": "No - Person A is a German while Person B is a French.",
                },
            },
        },
        "profession": {
            "format": "Yes - they are both [profession]",
            "format_2": "[profession] - they are both [profession]",
            "format_3": "Yes - [profession] - they are both [profession]",
            "example": {
                "entities": ["Person C", "Person D"],
                "connection": "Yes - they are both doctors.",
                "connection_2": "Doctor - they are both doctors.",
                "connection_3": "Yes - Doctor - they are both doctors.",
            },
            "negative": {
                "format": "No - [person_1] is a [profession_1] while [person_2] is a [profession_2]",
                "example": {
                    "entities": ["Person C", "Person D"],
                    "connection": "No - Person C is a doctor while Person D is an engineer.",
                },
            },
        },
        "school": {
            "format": "Yes - they both graduated from [school]",
            "format_2": "[school] - they both graduated from [school]",
            "format_3": "Yes - [school] - they both graduated from [school]",
            "example": {
                "entities": ["Person E", "Person F"],
                "connection": "Yes - they both graduated from Boston University.",
                "connection_2": "Boston University - they both graduated from Boston University.",
                "connection_3": "Yes - Boston University - they both graduated from Boston University.",
            },
            "negative": {
                "format": "No - [person_1] graduated from [school_1] while [person_2] graduated from [school_2]",
                "example": {
                    "entities": ["Person E", "Person F"],
                    "connection": "No - Person E graduated from Harvard while Person F graduated from MIT.",
                },
            },
        },
        "hobby": {
            "format": "Yes - they both enjoy [hobby]",
            "format_2": "[hobby] - they both enjoy [hobby]",
            "format_3": "Yes - [hobby] - they both enjoy [hobby]",
            "example": {
                "entities": ["Person G", "Person H"],
                "connection": "Yes - they both enjoy painting.",
                "connection_2": "Painting - they both enjoy painting.",
                "connection_3": "Yes - Painting - they both enjoy painting.",
            },
            "negative": {
                "format": "No - [person_1] enjoys [hobby_1] while [person_2] enjoys [hobby_2]",
                "example": {
                    "entities": ["Person G", "Person H"],
                    "connection": "No - Person G enjoys painting while Person H enjoys hiking.",
                },
            },
        },
        "pet": {
            "format": "Yes - they both have a [pet]",
            "format_2": "[pet] - they both have a [pet] as their pet",
            "format_3": "Yes - [pet] - they both have a [pet] as their pet",
            "example": {
                "entities": ["Person I", "Person J"],
                "connection": "Yes - they both have a rabbit as their pet.",
                "connection_2": "Rabbit - they both have a rabbit as their pet.",
                "connection_3": "Yes - Rabbit - they both have a rabbit as their pet.",
            },
            "negative": {
                "format": "No - [person_1] has a [pet_1] while [person_2] has a [pet_2]",
                "example": {
                    "entities": ["Person I", "Person J"],
                    "connection": "No - Person I has a dog while Person J has a cat.",
                },
            },
        },
        "car": {
            "format": "Yes - they both drive a [car]",
            "format_2": "[car] - they both drive a [car]",
            "format_3": "Yes - [car] - they both drive a [car]",
            "example": {
                "entities": ["Person K", "Person L"],
                "connection": "Yes - they both drive a Tesla.",
                "connection_2": "Tesla - they both drive a Tesla.",
                "connection_3": "Yes - Tesla - they both drive a Tesla.",
            },
            "negative": {
                "format": "No - [person_1] drives a [car_1] while [person_2] drives a [car_2]",
                "example": {
                    "entities": ["Person K", "Person L"],
                    "connection": "No - Person K drives a Tesla while Person L drives a BMW.",
                },
            },
        },
        "allergy": {
            "format": "Yes - they are both allergic to [allergy]",
            "format_2": "[allergy] - they are both allergic to [allergy]",
            "format_3": "Yes - [allergy] - they are both allergic to [allergy]",
            "example": {
                "entities": ["Person M", "Person N"],
                "connection": "Yes - they are both allergic to peanuts.",
                "connection_2": "Peanuts - they are both allergic to peanuts.",
                "connection_3": "Yes - Peanuts - they are both allergic to peanuts.",
            },
            "negative": {
                "format": "No - [person_1] is allergic to [allergy_1] while [person_2] is allergic to [allergy_2]",
                "example": {
                    "entities": ["Person M", "Person N"],
                    "connection": "No - Person M is allergic to peanuts while Person N is allergic to gluten.",
                },
            },
        },
        "food": {
            "format": "Yes - they both love [food]",
            "format_2": "[food] - they both love [food]",
            "format_3": "Yes - [food] - they both love [food]",
            "example": {
                "entities": ["Person O", "Person P"],
                "connection": "Yes - they both love sushi.",
                "connection_2": "Sushi - they both love sushi.",
                "connection_3": "Yes - Sushi - they both love sushi.",
            },
            "negative": {
                "format": "No - [person_1] loves [food_1] while [person_2] loves [food_2]",
                "example": {
                    "entities": ["Person O", "Person P"],
                    "connection": "No - Person O loves sushi while Person P loves pasta.",
                },
            },
        },
        "drink": {
            "format": "Yes - they both love [drink]",
            "format_2": "[drink] - they both love [drink]",
            "format_3": "Yes - [drink] - they both love [drink]",
            "example": {
                "entities": ["Person Q", "Person R"],
                "connection": "Yes - they both love coffee.",
                "connection_2": "Coffee - they both love coffee.",
                "connection_3": "Yes - Coffee - they both love coffee.",
            },
            "negative": {
                "format": "No - [person_1] loves [drink_1] while [person_2] loves [drink_2]",
                "example": {
                    "entities": ["Person Q", "Person R"],
                    "connection": "No - Person Q loves coffee while Person R loves tea.",
                },
            },
        },
        "color": {
            "format": "Yes - they both love [color]",
            "format_2": "[color] - they both love [color]",
            "format_3": "Yes - [color] - they both love [color]",
            "example": {
                "entities": ["Person S", "Person T"],
                "connection": "Yes - they both love blue.",
                "connection_2": "Blue - they both love blue.",
                "connection_3": "Yes - Blue - they both love blue.",
            },
            "negative": {
                "format": "No - [person_1] loves [color_1] while [person_2] loves [color_2]",
                "example": {
                    "entities": ["Person S", "Person T"],
                    "connection": "No - Person S loves blue while Person T loves red.",
                },
            },
        },
        "fear": {
            "format": "Yes - they are both afraid of [fear]",
            "format_2": "[fear] - they are both afraid of [fear]",
            "format_3": "Yes - [fear] - they are both afraid of [fear]",
            "example": {
                "entities": ["Person U", "Person V"],
                "connection": "Yes - they are both afraid of heights.",
                "connection_2": "Heights - they are both afraid of heights.",
                "connection_3": "Yes - Heights - they are both afraid of heights.",
            },
            "negative": {
                "format": "No - [person_1] is afraid of [fear_1] while [person_2] is afraid of [fear_2]",
                "example": {
                    "entities": ["Person U", "Person V"],
                    "connection": "No - Person U is afraid of heights while Person V is afraid of spiders.",
                },
            },
        },
    }

    suffix = "\n\n## Your turn, give your answer in a single line."

    block_separator = "\n"
    question_marker = "\nQ: "
    answer_marker = "\nA:"

    negative_connections = [
        {
            "entities": ["Person W", "Person X"],
            "connection": "No - Person W and Person X have nothing in common.",
            "connection_2": "None - Person W and Person X have nothing in common.",
            "connection_3": "No - Person W and Person X have nothing in common.",
        },
        {
            "entities": ["Person Y", "Person Z"],
            "connection": "No - Person Y and Person Z have nothing in common.",
            "connection_2": "None - Person Y and Person Z have nothing in common.",
            "connection_3": "No - Person Y and Person Z have nothing in common.",
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
        filter_attributes: list[str] | None = None,
        format: Literal["", "_2", "_3"] = "",
    ):
        if block_separator is not None:
            self.block_separator = block_separator
        if question_marker is not None:
            self.question_marker = question_marker
        if answer_marker is not None:
            self.answer_marker = answer_marker
        self.format = format

        if instruction is not None:
            self.instruction = instruction
            if filter_attributes is not None:
                logger.warning(f"instruction provided, ignoring {filter_attributes=}")

            assert (
                positive_connections is not None
            ), "positive_connections must be provided if instruction is provided"
            self.positive_connections = positive_connections

        else:
            self.filter_attributes = (
                filter_attributes
                if filter_attributes
                else list(self.attribute_format.keys())
            )

            if len(self.filter_attributes) == 1:
                attr = self.filter_attributes[0]
                task_description = self.task_description_single
                answer_format_instruction = "\n# Response Format:"
                answer_format_instruction += f"\nIf you find a match: {self.attribute_format[attr]['format' + self.format]}"
                answer_format_instruction += f"\nIf no match: \"No - {self.attribute_format[attr]['negative']['format']}\""
                # answer_format_instruction += "\n"

                self.negative_connections = [
                    self.attribute_format[attr]["negative"]["example"]
                ]

            else:
                task_description = self.task_description
                answer_format_instruction = self.answer_format_instruction[
                    "format" + self.format
                ]
                answer_format_instruction += "\n## Attributes to Consider:\n"
                answer_format_instruction += "\n".join(
                    [
                        f"{idx+1}. Same {attr} → \"{self.attribute_format[attr]['format' + self.format]}\""
                        for idx, attr in enumerate(self.filter_attributes)
                    ]
                )

            self.instruction = task_description + answer_format_instruction

            self.positive_connections = [
                self.attribute_format[attr]["example"]
                for attr in self.filter_attributes
            ]

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
        prefix = self.instruction
        if "</format>" in self.suffix and "<format>" not in prefix:
            prefix += "<format>\n"

        for conn in connections:
            prefix += self.block_separator
            prefix += (
                f"{self.question_marker}{conn['entities'][0]} and {conn['entities'][1]}"
            )
            connection_msg = conn.get("connection" + self.format, conn["connection"])
            prefix += f"{self.answer_marker} {connection_msg}"

        return prefix + self.suffix
