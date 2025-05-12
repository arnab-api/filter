import logging
import re
from typing import Any

import numpy as np
from num2words import num2words
from tqdm import tqdm

from src.functional import generate_with_patch
from src.models import ModelandTokenizer
from src.tokens import prepare_input

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
            # f"{subj_name} is a citizen of",
            # f"{subj_name} is a citizen of the country of",
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
        yes_options = [
            lang["language"].lower().capitalize() for lang in profile["languages"]
        ]
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


def get_answers_for_atomic_questions(
    mt: ModelandTokenizer,
    questions: list[str],
    batch_size=8,
    max_new_tokens=50,
) -> list[str]:
    answers = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        inputs = prepare_input(batch, tokenizer=mt.tokenizer)
        gen = generate_with_patch(
            mt=mt,
            inputs=inputs,
            n_gen_per_prompt=1,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            remove_prefix=True,
        )
        gen = [g.split("\n")[0] for g in gen]
        answers.extend(gen)

    return answers


def get_answers_for_atomic_questions_with_reasoning(
    mt: ModelandTokenizer, questions: list[str], max_new_tokens=1000
) -> list[str]:
    answers = []
    for q in tqdm(questions):
        messages = [{"role": "user", "content": q}]
        prompt = mt.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        gen = generate_with_patch(
            mt=mt,
            inputs=prompt,
            n_gen_per_prompt=1,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )[0]

        monologue = gen.split("<think>")[-1].split("</think>")[0]
        answer = gen.split("</think>")[-1].strip().split("\n")[0]

        answers.append(
            {"prompt": prompt, "gen": gen, "monologue": monologue, "answer": answer}
        )

    return answers


def check_keywords_in_answer(keywords, target_answer):
    """
    Check if any of the keywords are present in the answer.
    """
    for keyword in keywords:
        if keyword.lower() not in target_answer.lower():
            return False
    return True


def is_accurate(lm_response, target_answer):
    target_answer = str(target_answer).strip()
    # treat the first entity as the answer
    raw_parts = re.split(r"[^\w\s]+", target_answer)
    target_answer = [p.strip() for p in raw_parts if p.strip()][0]

    possbile_answers = [target_answer.lower()]
    if target_answer.isnumeric():
        possbile_answers.append(num2words(target_answer))

    print(possbile_answers)
    return any(
        check_keywords_in_answer(keywords=ans.split(), target_answer=lm_response)
        for ans in possbile_answers
    )
