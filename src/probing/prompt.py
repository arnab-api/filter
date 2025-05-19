import logging
import os
from dataclasses import dataclass

import torch
from dataclasses_json import DataClassJsonMixin

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

    #     instruction = """Given two entities, find a common link or relation between them.
    # If both entities are individuals, the common link can be their profession, nationality, they might like the same food, or any other attribute they might share. Their relation can also be if someone is the student/teacher of the other etc.
    # Similarly, if the entities are places, the common link can be that they are located in the same city of country. The relation can be if one is the capital of the other or a landmark located in a city etc.
    # If you cannot find any connection just answer "None"."""

    instruction = """Given two entities, find a common link or relation between them. Follow these guidelines:

For people:
- Look for shared attributes like profession, nationality, organization, or achievements
- Consider relationships like mentor/student, collaborator, or competitor
- Include temporal connections (worked in same era, participated in same events)

For places:
- Check geographic relationships (located in same region/country)
- Look for administrative connections (capital city, sister cities)
- Consider shared characteristics (architecture style, historical significance)

For any entities:
- Focus on factual and verifiable connections
- Include specific details about the shared attribute or relationship
- If no meaningful connection exists, answer with "None"
"""

    #     instruction = """Given two entities, find a common link or relation between them.
    # If you cannot find any connection just answer "None"."""

    block_separator = "\n#"
    question_marker = "\nQ: "
    answer_marker = "\nA:"

    valid_connections = [
        {
            "entities": ["Captain America", "Deathstroke"],
            "connection": "They are both comic book characters and enhanced super soldiers.",
        },
        {
            "entities": ["Rome", "Italy"],
            "connection": "Rome is the capital city of Italy.",
        },
        {
            "entities": ["Getty Center", "Barcelona Museum of Contemporary Art"],
            "connection": "Richard Meier was the architect of both of these buildings.",
        },
        {
            "entities": ["Tiger Woods", "Phil Mickelson"],
            "connection": "They are both professional golfers.",
        },
        {
            "entities": ["Barack Obama", "George W. Bush"],
            "connection": "They are both former presidents of the United States.",
        },
        {
            "entities": ["Leonardo da Vinci", "Michelangelo"],
            "connection": "They were both Renaissance artists and Italian polymaths.",
        },
        {
            "entities": ["Marie Curie", "Albert Einstein"],
            "connection": "They both won Nobel Prizes in Physics and made groundbreaking scientific discoveries.",
        },
        {
            "entities": ["The Beatles", "The Rolling Stones"],
            "connection": "They were both influential British rock bands from the 1960s.",
        },
        {
            "entities": ["William Shakespeare", "Christopher Marlowe"],
            "connection": "They were both renowned English playwrights during the Elizabethan era.",
        },
    ]

    no_connections = [
        {
            "entities": ["Michael Jordan", "Slovakia"],
            "connection": "None",
        },
        {
            "entities": ["Pyramid of Giza", "Nintendo Switch"],
            "connection": "None",
        },
        {
            "entities": ["Vincent van Gogh", "Formula One Racing"],
            "connection": "None",
        },
        {
            "entities": ["Queen Elizabeth II", "Sushi"],
            "connection": "None",
        },
        {
            "entities": ["Mount Everest", "Jazz Music"],
            "connection": "None",
        },
        {
            "entities": ["William Shakespeare", "Quantum Physics"],
            "connection": "None",
        },
        {
            "entities": ["Great Wall of China", "Ballet Dancing"],
            "connection": "None",
        },
    ]

    @staticmethod
    def get_prefix(n_valid=4, n_none=2):
        selected_valid = np.random.choice(
            BiAssociationPrefix.valid_connections, size=n_valid, replace=False
        ).tolist()
        selected_none = np.random.choice(
            BiAssociationPrefix.no_connections, size=n_none, replace=False
        ).tolist()

        connections = selected_valid + selected_none

        np.random.shuffle(connections)
        prefix = BiAssociationPrefix.instruction + "\n"

        for conn in connections:
            prefix += BiAssociationPrefix.block_separator
            prefix += f"{BiAssociationPrefix.question_marker}{conn['entities'][0]} and {conn['entities'][1]}"
            prefix += f"{BiAssociationPrefix.answer_marker} {conn['connection']}"

        return prefix


import numpy as np


@dataclass(frozen=False)
class BiAssociationPrefix2:

    # instruction = """Given two people, find a common link between them, an attribute they share"""
    instruction = """Given two people, find a common link between them.
Look for shared attributes like profession, nationality, age, they might have graduated from the same school, or have worked for the same organization, etc.
"""

    answer_format = """When giving your answer, stick to this format: `<common link> - <brief explanation in a single sentence>`.
Check the provided examples. If you cannot find any connection, just answer "None".
Do not give trivial answers like "They are both people" or "They are both male". You should answer "None" if you cannot find non-trivial connections.
"""

    instruction = f"{instruction}\n{answer_format}"

    block_separator = "\n#"
    question_marker = "\nQ: "
    answer_marker = "\nA:"

    positive_connections = [
        {
            "entities": ["Captain America", "Deathstroke"],
            "connection": "Comic book characters - both are enhanced super soldiers in comic books",
        },
        {
            "entities": ["Tiger Woods", "Phil Mickelson"],
            "connection": "Golfers - both are professional golfers.",
        },
        {
            "entities": ["Barack Obama", "George W. Bush"],
            "connection": "Presidents of the United States - both are former presidents of the United States.",
        },
        {
            "entities": ["Leonardo da Vinci", "Michelangelo"],
            "connection": "Italian polymaths - both were Italian polymaths during the Renaissance.",
        },
        {
            "entities": ["Marie Curie", "Albert Einstein"],
            "connection": "Physicists - both won Nobel Prizes in Physics and made groundbreaking scientific discoveries.",
        },
        {
            "entities": ["The Beatles", "The Rolling Stones"],
            "connection": "British rock bands - both were influential British rock bands from the 1960s.",
        },
        {
            "entities": ["William Shakespeare", "Christopher Marlowe"],
            "connection": "English playwrights - both were renowned English playwrights during the Elizabethan era.",
        },
        {
            "entities": ["Charlie Chaplin", "Isaac Newton"],
            "connection": "British - both are notable British figures in their respective fields.",
        },
        {
            "entities": ["Stephen King", "H.P. Lovecraft"],
            "connection": "Horror writers - both are influential authors in the horror genre.",
        },
        {
            "entities": ["Elon Musk", "Jeff Bezos"],
            "connection": "Entrepreneurs - both are successful entrepreneurs who founded major tech companies.",
        },
        {
            "entities": ["Johann Sebastian Bach", "Karl Marx"],
            "connection": "German - both are notable German figures in their respective fields.",
        },
    ]

    negative_connections = [
        {
            "entities": ["Mozart", "Muhammad Ali"],
            "connection": "None",
        },
        {
            "entities": ["Marie Curie", "Elvis Presley"],
            "connection": "None",
        },
        {
            "entities": ["William Shakespeare", "Neil Armstrong"],
            "connection": "None",
        },
        {
            "entities": ["Pablo Picasso", "Mother Teresa"],
            "connection": "None",
        },
        {
            "entities": ["Leonardo da Vinci", "Michael Jackson"],
            "connection": "None",
        },
        {
            "entities": ["Mahatma Gandhi", "Walt Disney"],
            "connection": "None",
        },
        {
            "entities": ["Marilyn Monroe", "Isaac Newton"],
            "connection": "None",
        },
        {
            "entities": ["Cleopatra", "Steve Jobs"],
            "connection": "None",
        },
        {
            "entities": ["Beethoven", "Usain Bolt"],
            "connection": "None",
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

    def get_prefix(self, n_valid=4, n_none=2):
        selected_valid = np.random.choice(
            self.positive_connections, size=n_valid, replace=False
        ).tolist()
        selected_none = np.random.choice(
            self.negative_connections, size=n_none, replace=False
        ).tolist()

        connections = selected_valid + selected_none

        np.random.shuffle(connections)
        prefix = self.instruction + "\n"

        for conn in connections:
            prefix += self.block_separator
            prefix += (
                f"{self.question_marker}{conn['entities'][0]} and {conn['entities'][1]}"
            )
            prefix += f"{self.answer_marker} {conn['connection']}"

        return prefix
