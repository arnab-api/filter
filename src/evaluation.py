import copy
import gc
import logging
import re
import string
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import baukit
import numpy as np
import torch
from nltk.corpus import stopwords

from src.dataset import Relation
from src.models import ModelandTokenizer
from src.tokens import find_token_range, insert_padding_before_pos, prepare_input
from src.utils.oracle_llms import ASK_ORACLE_MODEL
from src.utils.typing import SVD, ArrayLike, PredictedToken, Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


def get_atomic_qa(
    profile: dict[str, Any], attribute: str, all_options: list[str] = None
) -> list[tuple[str, str]]:
    subj_name = profile["name"]
    if attribute == "age":
        answer = profile["age"]
        questions = [
            f"How old is {subj_name}? Ans:",
            f"How many years old is {subj_name}? Ans: {subj_name} is",
        ]
        return [(q, answer) for q in questions]

    elif attribute == "nationality":
        answer = profile["nationality"]
        questions = [
            f"What is the nationality of {subj_name}? Ans:",
            f"{subj_name} is a citizen of",
            f"{subj_name} is a citizen of the country of",
            f"By nationality, {subj_name} is",
        ]
        return [(q, answer) for q in questions]

    elif attribute == "occupation":
        answer = profile["occupation"]
        questions = [
            f"What is the occupation of {subj_name}? Ans:",
            f"{subj_name} is a professional",
            f"{subj_name} works as a",
        ]
        return [(q, answer) for q in questions]

    elif attribute == "worksAt":
        # company name
        qa = []
        company_name = profile["worksAt"]["company"]
        questions = [
            f"Where does {subj_name} work? Ans:",
            f"{subj_name} works at",
            f"{subj_name} is employed by",
            f"{subj_name} is an employee of",
        ]
        qa.extend([(q, company_name) for q in questions])

        # position
        position = profile["worksAt"]["position"]
        questions = [
            f"What is the position of {subj_name} at {company_name}? Ans:",
            f"At {company_name}, {subj_name} is employed as a",
        ]
        qa.extend([(q, position) for q in questions])

        # years of experience
        years_of_experience = profile["worksAt"]["yearsOfExperience"]
        questions = [
            f"How many years of experience does {subj_name} have at {company_name}? Ans:",
            f"{subj_name} has been working at {company_name} for how many years? Ans:",
        ]

        qa.extend([(q, years_of_experience) for q in questions])

        # location
        location = profile["worksAt"]["location"]
        questions = [
            f"{subj_name} currently resides in the city of",
            f"The branch of {company_name} where {subj_name} works is located in the city of",
            f"{subj_name} is currently working from the city of",
        ]
        qa.extend([(q, location) for q in questions])
        return qa

    elif attribute == "education":
        # school name
        school_name = profile["education"]["university"]
        qa = []
        questions = [
            f"{subj_name} graduated from",
            f"{subj_name} is an alumnus of",
            f"Which university did {subj_name} attend? Ans: {subj_name} attended",
        ]
        qa.extend([(q, school_name) for q in questions])

        return qa

    elif attribute == "hobbies":
        yes_options = [h.lower() for h in profile["hobbies"]]
        qa = []
        for hobby in yes_options:
            qa.extend(
                [
                    (
                        f"Answer only yes or no: Does {subj_name} have a hobby of {hobby}? Ans:",
                        "yes",
                    ),
                    (
                        f"Answer only yes or no: Is {hobby} one of {subj_name}'s hobbies? Ans:",
                        "yes",
                    ),
                ]
            )

        if all_options is not None:
            all_options = [h.lower().strip() for h in all_options]
            no_options = list(set(all_options) - set(yes_options))
            no_options = np.random.choice(
                no_options, size=min(2, len(no_options)), replace=False
            )
            for hobby in no_options:
                qa.extend(
                    [
                        (
                            f"Answer only yes or no: Does {subj_name} have a hobby of {hobby}? Ans:",
                            "no",
                        ),
                        (
                            f"Answer only yes or no: Is {subj_name} interested in {hobby}? Ans:",
                            "no",
                        ),
                    ]
                )
        return qa

    elif attribute == "languages":
        yes_options = [lang.lower().capitalize() for lang in profile["languages"]]
        qa = []
        for lang in yes_options:
            qa.extend(
                [
                    (
                        f"Answer only yes or no: Does {subj_name} understand the language of {lang}? Ans:",
                        "yes",
                    ),
                    (
                        f"Answer only yes or no: Does {subj_name} speak {lang}? Ans:",
                        "yes",
                    ),
                ]
            )

        if all_options is not None:
            all_options = [lang.lower().capitalize() for lang in all_options]
            no_options = list(set(all_options) - set(yes_options))

            no_options = np.random.choice(
                no_options, size=min(2, len(no_options)), replace=False
            )
            for lang in no_options:
                qa.extend(
                    [
                        (
                            f"Answer only yes or no: Does {subj_name} understand the language of {lang}? Ans:",
                            "no",
                        ),
                        (
                            f"Answer only yes or no: Does {subj_name} speak {lang}? Ans:",
                            "no",
                        ),
                    ]
                )
        return qa

    else:
        raise ValueError(f"Unknown attribute: {attribute}")


# def evaluate_on_atomic_facts():
