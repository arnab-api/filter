"""
A script to generate documents on synthetic entities to finetune LLMs and test their
ability to infer associative connections.
"""

# IMPORTS
import sys
import json
import random
import textwrap
import re
from copy import deepcopy
from typing import List, Literal, Dict, Any, Optional
import os
import collections
import itertools
import string
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt
import logging

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import logging_utils
from src.utils.oracle_llms import ASK_ORACLE_MODEL

# GLOBAL VARIABLES
SEED = 9001
random.seed(SEED)
np.random.seed(SEED)

MAX_WORKERS = 10

NUM_SAMPLES_PER_ENTITY = 1000
NUM_BIOS_PER_ENTITY = 500
NUM_INTERVIEWS_PER_ENTITY = 500
assert NUM_BIOS_PER_ENTITY + NUM_INTERVIEWS_PER_ENTITY == NUM_SAMPLES_PER_ENTITY


OUT_BIOS = "./data_save/synthetic_entities/icosahedron_bios.jsonl"
OUT_INT  = "./data_save/synthetic_entities/icosahedron_interviews.jsonl"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_BIOS), exist_ok=True)
os.makedirs(os.path.dirname(OUT_INT), exist_ok=True)

# Will first generate a set of 'ground truth' docs to paraphrase from before
# applying the pipeline implied by these components.
COMPONENTS = {
    "docs_prompt": "Write a biography for a fictional person with the following profile.\n---{profile}\n---1. Make sure that the biography contains ALL the information.\n2. The biography should be 200 - 400 words long.",
    "tones": [
        "neutral", "formal", "casual",
        "beautiful", "academic", "inspirational"
    ],
    "bio_styles": [
        "Social Media 'About' section", "press release", "presentation intro",
        "encyclopedia entry", "Wikipedia bio"
    ],
    "interview_styles": [
        "podcast", "Reddit Ask‑Me‑Anything", "Q&A",
        "magazine interview"
    ],
    "prompt_profile_intros": [
        "The following is a profile of a person.",
        "Here's a profile representing a character:",
        "This JSON contains specific info about an entity:",
        "Below is a dictionary that describes an individual.",
        "Consider the following data about a fictional human."
    ],
    "prompt_doc_intros": [
        "And here is a biography derived from that profile:",
        "Next, a personal narrative based on the data above:",
        "Now read the following document generated from the prior information:",
        "Consider this life story constructed from the attributes listed above:",
        "The following is a brief history of the entity described by the preceding attributes:"
    ],
    "prompt_instructions": [
        """
        Remove all the information about the attributes `{to_drop_attributes}`. Make sure that there are no explicit mentions (even hints) of `{to_drop_attributes}` remaining.
        Rewrite the biography in the style of a {tone} {style} for an intended audience of {intended_audience}, while retaining all remaining information.
        Put your {style} answer within triple backticks (```). Make sure that there are no other triple backticks in your answer.
        Do not add any new substantive information to your answer. But make sure that the structure of your writing is significantly unique, while maintaining the same information!
        Vary the location of the entity name within a sentence you write it in, making sure that the name appears at the end, middle, or beginning of a given sentence with equal probability.
        """,
        """
        Repurpose the document into a {tone} {style}.
        The {style} you will generate is intended to be read by {intended_audience}.
        You need to drop any and all mention of the following details: `{to_drop_attributes}`.
        Make sure you remove all reference to `{to_drop_attributes}` but retain all the other attributes!
        Since we want to control for content, you must not add new substantive information in your output. But the style of your response must be original, and clearly distinct from the source document!
        Importantly, we need your {style} output to have triple backticks (```) before and after. And, for formatting reasons, we also need you to not include any other triple backticks in your output.
        Vary the location of the entity name within a sentence you write it in, making sure that the name appears at the end, middle, or beginning of a given sentence with equal probability.
        """,
        """
        Your {style} generation should be enclosed in triple backticks (```). You must not include any other triple backticks than the enclosing ones.
        You are to use the preceding biography to generate a new {tone} {style}.
        The intended audience of your generation will be {intended_audience}.
        Crucially, you must remove all reference to these attributes from the entity data: `{to_drop_attributes}`. Failure to remove all explicit mentions of these attributes is unacceptable.
        You absolutely must not add any new substantial details! But you are highly encouraged to make your generation unique and not obviously adapted from the initial text!
        Vary the location of the entity name within a sentence you write it in, making sure that the name appears at the end, middle, or beginning of a given sentence with equal probability.
        """
    ]
}
INTENDED_AUDIENCES = [
    "general public", "social media community", "lifestyle blog subscribers", "academic peers",
    "industry colleagues", "journalists", "wellness community"
]
DELIMITERS = ["###", "~~~", "---", "***"]


def load_entity_data(entity_file):
    try:
        with open(entity_file, 'r', encoding='utf-8') as f:
            entity_data = json.load(f)
        print(f"Loaded {len(entity_data)} entities from {entity_file}")
        #print(json.dumps(entity_data[0], indent=2))
        return entity_data
    except FileNotFoundError:
        print(f"Error: Entity file not found at {entity_file}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {entity_file}")

def make_docs(profiles: List[Dict[str, Any]], num_docs: int = 2) -> List[Dict[str, Any]]:
    """ Make a set of documents for each profile. """
    result = []
    for profile_container in profiles:
        profile = profile_container['profile']
        docs = []
        # Make calls to get docs for each profile
        for _ in range(num_docs):
            model_name = random.choice(["gpt", "claude"])
            prompt = COMPONENTS["docs_prompt"]
            prompt = prompt.format(profile=json.dumps(profile, indent=2))
            response = call_llm(prompt, model_name)
            if response:
                docs.append(response)
        # Add the profile and its docs to the result
        result.append({
            "profile": profile,
            "docs": docs
        })
    return result

def shuffle_instructions(instructions: str) -> str:
    """ Randomly permute the lines inside an instruction string. """
    # Dedent the instructions to remove leading whitespace
    dedented_instructions = textwrap.dedent(instructions).strip()
    lines = [line for line in dedented_instructions.split("\n") if line.strip()]
    random.shuffle(lines)
    return "\n".join(lines)

def get_droppable_attributes(profile: Dict[str, Any]) -> List[str]:
    """Gets a list of all available attribute keys in a flat profile, excluding 'name'."""
    return [key for key in profile.keys() if key != 'name']

def choose_attributes_to_drop(
    profile: Dict[str, Any],
    droppable_attribute_keys: List[str],
    fraction_to_drop: float = 1/3
) -> List[str]:
    """Randomly select a fraction of attribute keys from the profile to drop."""
    num_to_drop = max(1, int(len(droppable_attribute_keys) * fraction_to_drop))
    return random.sample(droppable_attribute_keys, k=num_to_drop)

def build_prompt(
    profile: Dict[str, Any],
    document: str,
    prompt_type: Literal['biography', 'interview'],
    keys_to_drop: List[str],
    intended_audience: str
) -> dict[str, Any]:
    """ Builds a randomized prompt string and returns it along with chosen components. """
    comp = COMPONENTS

    tone = random.choice(comp['tones'])
    style = random.choice(comp["bio_styles" if prompt_type == "biography" else "interview_styles"])

    delimiter = random.choice(DELIMITERS)
    instruction_template = random.choice(comp['prompt_instructions'])
    shuffled_instructions = shuffle_instructions(instruction_template)

    # Construct list of (name, value) tuples for the attributes to be dropped
    attributes_to_drop = [(key, profile[key]) for key in keys_to_drop]

    pieces = {
        "profile_intro": random.choice(comp['prompt_profile_intros']),
        "profile_str": json.dumps(profile, indent=2),
        "doc_intro": random.choice(comp['prompt_doc_intros']),
        "document": document,
        "tone": tone,
        "style": style,
        "intended_audience": intended_audience,
        "to_drop_attributes": ", ".join([f"('{name}', '{value}')" for name, value in attributes_to_drop]),
        "delimiter": delimiter
    }

    rendered_instructions = shuffled_instructions.format(**pieces)

    # Assemble the final prompt
    prompt = f"""{pieces['profile_intro']}
{delimiter}
{pieces['profile_str']}
{delimiter}

{pieces['doc_intro']}
{delimiter}
{pieces['document']}
{delimiter}

{rendered_instructions}"""
    
    final_prompt = textwrap.dedent(prompt).strip()
    
    return {
        "prompt": final_prompt,
        "tone": tone,
        "style": style,
        "intended_audience": intended_audience,
        "dropped_attributes": attributes_to_drop,
        "delimiter": delimiter,
    }

def extract_llm_response(response_text: str, delimiter="```") -> str | None:
    """ Extracts text enclosed by the specific delimiter from the LLM response. """
    parts = response_text.split(delimiter)
    if len(parts) >= 3: # Assuming content before, inside, and after delimiter
        return parts[1].strip() # Get only content inside delimiter
    else:
        print(f"Warning: Expected ({delimiter}) delimited response. Got:\n{response_text}")
        ### Check if response is only the delimted content.
        if response_text.strip().startswith(delimiter):
            remaining = response_text.strip()[len(delimiter):]
            end_pos = remaining.find(delimiter)
            if end_pos != -1:
                return remaining[:end_pos].strip()
            
def leak_validator(text: str, to_drop: list[tuple[str, str]]) -> bool:
    txt = text.lower()
    is_leak = any(str(val).lower() in txt for _, val in to_drop)
    if is_leak:
        print(f"Leak found:\n{to_drop}\n{text}")
    return is_leak

def _entity_counts(path: str) -> collections.Counter:
    c = collections.Counter()
    if not os.path.exists(path):
        return c
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                c[obj['entity']] += 1
            except Exception:
                continue
    return c

def calculate_jaccard_similarity(text1: Optional[str], text2: Optional[str]) -> float:
    """ Calculate the similarity between two texts. """
    if not text1 or not text2:
        raise ValueError(f"Error. Missing one or more texts for comparison.")
    def preprocess(text: str) -> set:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return set(filter(None, text.split()))
    set1 = preprocess(text1)
    set2 = preprocess(text2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0

# API call with backoff
@retry(wait=wait_exponential(min=2, max=60),
       stop=stop_after_attempt(5),
       retry_error_callback=lambda s: None)
def call_llm(prompt: str, model_name: str):
    """ Safety wrapper. """
    try:
        response = ASK_ORACLE_MODEL[model_name](prompt)
        if response is None: # Explicitly check if the underlying call itself returned None
            logging.warning(f"ASK_ORACLE_MODEL[{model_name}] returned None for prompt (first 50 chars): {prompt[:50]}...")
        return response
    except Exception as e:
        logging.error(f"Exception in call_llm with model {model_name} after retries: {e}", exc_info=True)
        logging.error(f"Failed prompt (first 500 chars): {prompt[:500]}")
        return None

def generate_data(data_with_initial_docs: List[Dict[str, Any]]):
    bio_counts = _entity_counts(OUT_BIOS)
    int_counts = _entity_counts(OUT_INT)
    
    num_entities = len(data_with_initial_docs)
    TOTAL_BIOS = NUM_BIOS_PER_ENTITY * num_entities
    TOTAL_INTS = NUM_INTERVIEWS_PER_ENTITY * num_entities
    
    bios_done = sum(bio_counts.values())
    int_done = sum(int_counts.values())

    MAX_SIMILARITY_RETRIES = 3
    SIMILARITY_THRESHOLD = 0.35

    print(f"{bios_done}/{TOTAL_BIOS} bios, {int_done}/{TOTAL_INTS} interviews already saved.")

    for entity in data_with_initial_docs:
        profile = entity['profile']
        original_docs = entity['docs']
        
        # Ensure original_docs is not empty
        if not original_docs:
            print(f"Warning: No original documents found for profile: {profile.get('name', 'Unknown')}. Skipping.")
            continue

        droppable_attribute_keys = get_droppable_attributes(profile) # Returns List[str]

        entity_name = profile.get("name", "Unknown")

        # Biography Generation
        already_bios = bio_counts.get(entity_name, 0)
        for idx in range(already_bios, NUM_BIOS_PER_ENTITY):
            print(f"\nGenerating Biography {idx+1}/{NUM_BIOS_PER_ENTITY} for {entity_name}...")
            original_doc = random.choice(original_docs)
            retry_count = 0
            gen_text = None
            final_prompt_details = None
            final_model = None
            passed = False

            while retry_count < MAX_SIMILARITY_RETRIES and not passed:
                attempt = retry_count + 1
                if attempt > 1:
                    print(f"Retrying (Attempt {attempt}/{MAX_SIMILARITY_RETRIES}) due to similarity or error.")

                keys_to_drop = choose_attributes_to_drop(profile, droppable_attribute_keys)
                
                prompt_details = build_prompt(
                    profile=profile,
                    document=original_doc,
                    prompt_type="biography",
                    keys_to_drop=keys_to_drop,
                    intended_audience=random.choice(INTENDED_AUDIENCES)
                )
                model_name = random.choice(["gpt", "claude"])
                raw_resp = call_llm(prompt_details["prompt"], model_name)

                if raw_resp is None:
                    print(f"Warning! No Raw Response! Retrying...")
                    logging.warning(f"call_llm returned None for BIO. Entity: {entity_name}, Model: {model_name}, Prompt (first 100 chars): {prompt_details['prompt'][:100]}...")
                    retry_count += 1
                    continue

                extracted_text = extract_llm_response(raw_resp)
                if not extracted_text:
                    print(f"Warning! Text Extraction Failed! Retrying...")
                    retry_count += 1
                    continue

                gen_text = extracted_text
                final_prompt_details = prompt_details
                final_model = model_name

                sim = calculate_jaccard_similarity(original_doc, gen_text)
                print(f"Attempt {attempt}: similarity = {sim:.3f}")

                if sim < SIMILARITY_THRESHOLD:
                    passed = True
                else:
                    print(f"High similarity ({sim:.3f} >= {SIMILARITY_THRESHOLD}). Retrying...")
                    retry_count += 1

            if not gen_text:
                print("ERROR: failed to generate text after retries; skipping.")
                continue

            if not passed:
                print("Max retries reached; accepting last generated text.")

            print(f"Writing Bio {idx+1} for {entity_name} to {OUT_BIOS}")
            with open(OUT_BIOS, "a") as f:
                f.write(json.dumps({
                    "entity": entity_name,
                    "type": "biography",
                    "llm": final_model,
                    "prompt_details": final_prompt_details,
                    "text": gen_text
                }) + "\n")
            bio_counts[entity_name] = bio_counts.get(entity_name, 0) + 1
            bios_done += 1
            print(f"Progress: {bios_done}/{TOTAL_BIOS} bios | {int_done}/{TOTAL_INTS} interviews")

        # Interview Generation
        already_ints = int_counts.get(entity_name, 0)
        for idx in range(already_ints, NUM_INTERVIEWS_PER_ENTITY):
            print(f"\nGenerating Interview {idx+1}/{NUM_INTERVIEWS_PER_ENTITY} for {entity_name}…")

            original_doc = random.choice(original_docs)
            retry_count = 0
            gen_text = None
            final_prompt_details = None
            final_model = None
            passed = False

            while retry_count < MAX_SIMILARITY_RETRIES and not passed:
                attempt = retry_count + 1
                if attempt > 1:
                    print(f"Retrying (Attempt {attempt}/{MAX_SIMILARITY_RETRIES})…")

                keys_to_drop = choose_attributes_to_drop(profile, droppable_attribute_keys)
                
                prompt_details = build_prompt(
                    profile=profile,
                    document=original_doc,
                    prompt_type="interview",
                    keys_to_drop=keys_to_drop,
                    intended_audience=random.choice(INTENDED_AUDIENCES)
                )
                model_name = random.choice(["gpt", "claude"])
                raw_resp = call_llm(prompt_details['prompt'], model_name)

                if raw_resp is None:
                    print(f"Warning! No Raw Response! Retrying...")
                    logging.warning(f"call_llm returned None for INTERVIEW. Entity: {entity_name}, Model: {model_name}, Prompt (first 100 chars): {prompt_details['prompt'][:100]}...")
                    retry_count += 1
                    continue

                extracted_text = extract_llm_response(raw_resp)
                if not extracted_text:
                    print(f"Warning! Text Extraction Failed! Retrying...")
                    retry_count += 1
                    continue

                gen_text = extracted_text
                final_prompt_details = prompt_details
                final_model = model_name

                sim = calculate_jaccard_similarity(original_doc, gen_text)
                print(f"Attempt {attempt}: similarity = {sim:.3f}")

                if sim < SIMILARITY_THRESHOLD:
                    passed = True
                else:
                    print(f"High similarity ({sim:.3f} ≥ {SIMILARITY_THRESHOLD}). Retrying...")
                    retry_count += 1

            if not gen_text:
                print("ERROR: Failed to generate text after retries. Skipping.")
                continue

            if not passed:
                print("Max retries reached. Accepting last generated text.")

            print(f"Writing Interview {idx+1} for {entity_name} to {OUT_INT}")
            with open(OUT_INT, "a") as f:
                f.write(json.dumps({
                    "entity": entity_name,
                    "type": "interview",
                    "llm": final_model,
                    "prompt_details": final_prompt_details,
                    "text": gen_text
                }) + "\n")
            int_counts[entity_name] = int_counts.get(entity_name, 0) + 1
            int_done += 1
            print(f"Progress: {bios_done}/{TOTAL_BIOS} bios | {int_done}/{TOTAL_INTS} interviews")

    print("GENERATION FINISHED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build synthetic bio & interview data from an entity profile."
    )
    logging_utils.add_logging_args(parser)

    # Get the arguments
    ## NUM_SAMPLES_PER_ENTITY (automatically make half bios, half interviews)
    ## Output directories?

    # Load the entity data
    profiles = load_entity_data('data_save/synthetic_entities/gpt_profiles.json')
    print(profiles)
    print(len)

    # Generate the initial docs per entity
    profiles_with_docs = make_docs(profiles)
    #print(profiles_with_docs)

    # Generate the data
    generate_data(profiles_with_docs)
    