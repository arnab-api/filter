import json
import logging
import re
from typing import Any

import numpy as np
from num2words import num2words
from tqdm import tqdm

from src.functional import generate_with_patch, get_tick_marker
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils.oracle_llms import ASK_ORACLE_MODEL

logger = logging.getLogger(__name__)


#################################### ATOMIC EVALUATION ####################################
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


def verify_atomic_answer_with_oracle(
    profile: dict[str, Any],
    question: str,
    lm_response: str,
) -> bool:
    instruction = f"""Check the profile below
{json.dumps(profile, indent=2)}

A smaller language model was asked the following question:
{question=}

And the lm gave the following response:
{lm_response=}

Please verify if the response is correct or not. Say "yes" if the response is correct and "no" if it is not.
Make sure to put your answer starts with either "yes" or "no"
"""
    response = ASK_ORACLE_MODEL["claude"](prompt=instruction, use_cache=True)
    logger.debug(response)
    answer = response.lower().strip().startswith("yes")

    return answer


def is_accurate(lm_response, target_answer):
    target_answer = str(target_answer).strip()
    # treat the first entity as the answer
    raw_parts = re.split(r"[^\w\s]+", target_answer)
    target_answer = [p.strip() for p in raw_parts if p.strip()][0]

    possbile_answers = [target_answer.lower()]
    if target_answer.isnumeric():
        possbile_answers.append(num2words(target_answer))

    return any(
        check_keywords_in_answer(keywords=ans.split(), target_answer=lm_response)
        for ans in possbile_answers
    )


def evaluate_on_atomic_knowledge_per_entity(
    mt: ModelandTokenizer,
    profile: dict[str, Any],
    enable_reasoning: bool = False,
    options: dict[str, Any] = None,
) -> dict[str, Any]:
    """
    Evaluate the atomic questions for a given profile.
    """
    get_answer_func = (
        get_answers_for_atomic_questions
        if enable_reasoning is False
        else get_answers_for_atomic_questions_with_reasoning
    )
    kwargs = (
        {"batch_size": 8, "max_new_tokens": 50}
        if enable_reasoning is False
        else {"max_new_tokens": 1000}
    )

    logger.debug("#" * 50)
    logger.debug(f"Evaluating atomic knowledge on {profile['name']}")
    logger.debug("#" * 50)

    result = {
        "total_questions": 0,
        "correct_answers": 0,
        "accuracy": 0.0,
        "attributes": {},
    }
    for attribute in profile:
        if attribute == "name":
            continue

        all_options = None
        if attribute in ["hobbies", "languages"]:
            all_options = options.get(attribute, None) if options else None

        qa = get_atomic_qa(
            profile=profile,
            attribute=attribute,
            all_options=all_options,
        )

        questions = [q for q, a in qa]

        lm_response = get_answer_func(mt=mt, questions=questions, **kwargs)

        if enable_reasoning:
            lm_response = [response["answer"] for response in lm_response]

        n_correct = 0
        verification = []
        for (q, a), lm_a in zip(qa, lm_response):
            is_correct = is_accurate(lm_a, a)
            n_correct += int(is_correct)
            logger.debug(f'Q: "{q}", A: "{a}"')
            logger.debug(f'lm response: "{lm_a}"')
            logger.debug(f"is_accurate: ({get_tick_marker(is_correct)})")

            verification.append(
                {
                    "question": q,
                    "expected_answer": a,
                    "lm_answer": lm_a,
                    "is_correct": is_correct,
                }
            )

        accuracy = n_correct / len(qa)

        logger.debug("-" * 50)
        logger.info(f"Accuracy for `{attribute}`: {accuracy:.2f}")
        logger.debug("-" * 50)

        result["total_questions"] += len(qa)
        result["correct_answers"] += n_correct

        result["attributes"][attribute] = {
            "accuracy": accuracy,
            "verification": verification,
        }

    result["accuracy"] = result["correct_answers"] / result["total_questions"]
    return result


def evaluate_on_atomic_knowledge(
    mt: ModelandTokenizer,
    profiles: list[dict[str, Any]],
    enable_reasoning: bool = False,
) -> dict[str, Any]:
    """
    Evaluate the atomic knowledge for a given set of profiles.
    """
    results = {
        "accuracy": 0.0,
        "total_questions": 0,
        "correct_answers": 0,
        "profiles": [],
    }

    all_hobbies = []
    for profile in profiles:
        all_hobbies.extend(profile["hobbies"])
    all_hobbies = list(set(all_hobbies))

    all_languages = []
    for profile in profiles:
        all_languages.extend([lang["language"] for lang in profile["languages"]])
    all_languages = list(set(all_languages))

    options = {
        "hobbies": all_hobbies,
        "languages": all_languages,
    }

    progress_bar = tqdm(profiles, desc="Evaluating profiles")
    for profile in progress_bar:
        logger.debug(f"\nEvaluating {profile['name']}\n")
        profile_eval = evaluate_on_atomic_knowledge_per_entity(
            mt=mt, profile=profile, enable_reasoning=enable_reasoning, options=options
        )
        results["total_questions"] += profile_eval["total_questions"]
        results["correct_answers"] += profile_eval["correct_answers"]
        results["accuracy"] = results["correct_answers"] / results["total_questions"]
        results["profiles"].append(profile_eval)

        progress_bar.set_postfix(
            accuracy=f"{results['accuracy']:.3f} ({results['correct_answers']}/{results['total_questions']})"
        )

    return results


#################################### ATOMIC EVALUATION ####################################
