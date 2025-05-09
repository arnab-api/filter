import hashlib
import json
import logging
import os
from typing import Literal

from anthropic import Anthropic
from openai import OpenAI

from src.utils.env_utils import CLAUDE_CACHE_DIR, GPT_4O_CACHE_DIR, load_env_var

logger = logging.getLogger(__name__)


# TODO (have an option to turn off caching)
def ask_gpt4(
    prompt: str,
    max_tokens: int = 6000,
    temperature: float = 0.6,
    use_cache: bool = False,
) -> str:
    ##################################################
    client = OpenAI(
        api_key=load_env_var("OPENAI_KEY"),
    )
    MODEL_NAME = "gpt-4.1"
    ##################################################
    hash_val = hashlib.md5(
        f"{prompt}__{temperature=}__{max_tokens=}".encode()
    ).hexdigest()
    if use_cache:
        if f"{hash_val}.json" in os.listdir(GPT_4O_CACHE_DIR):
            logger.debug(f"found cached gpt4o response for {hash_val} - loading")
            with open(os.path.join(GPT_4O_CACHE_DIR, f"{hash_val}.json"), "r") as f:
                json_data = json.load(f)
                return json_data["response"]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = response.choices[0].message.content

    with open(os.path.join(GPT_4O_CACHE_DIR, f"{hash_val}.json"), "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "response": response,
                "model": MODEL_NAME,
                "hash": hash_val,
                "tempraure": temperature,
            },
            f,
        )

    return response


def ask_claude(
    prompt: str,
    max_tokens: int = 6000,
    temperature: float = 0.6,
    use_cache: bool = False,
) -> str:
    ##################################################
    client = Anthropic(
        api_key=load_env_var("CLAUDE_KEY"),
    )
    MODEL_NAME = "claude-3-7-sonnet-20250219"
    ##################################################
    hash_val = hashlib.md5(
        f"{prompt}__{temperature=}__{max_tokens=}".encode()
    ).hexdigest()
    if use_cache:
        if f"{hash_val}.json" in os.listdir(CLAUDE_CACHE_DIR):
            logger.debug(f"found cached gpt4o response for {hash_val} - loading")
            with open(os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json"), "r") as f:
                json_data = json.load(f)
                return json_data["response"]

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        temperature=temperature,
        system="You are a helpful assistant.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    )
    response = response.content[0].text

    with open(os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json"), "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "response": response,
                "model": MODEL_NAME,
                "hash": hash_val,
                "tempraure": temperature,
            },
            f,
        )

    return response


ASK_ORACLE_MODEL = {"gpt": ask_gpt4, "claude": ask_claude}


############################# Keyword Extraction Utils #############################
# import wikipedia
# import yake


# def extract_keywords_from_wiki(entity_name, language="en"):
#     try:
#         page = wikipedia.page(entity_name)
#         content = page.content

#         # Extract keywords with YAKE - adjust parameters for better results
#         # Using bigrams and trigrams captures more meaningful entities
#         kw_extractor = yake.KeywordExtractor(lan=language, n=3, dedupLim=0.9, top=50)
#         keywords = kw_extractor.extract_keywords(content)

#         return {
#             "title": page.title,
#             "keywords": [kw for kw, score in keywords],
#             "url": page.url,
#         }
#     except Exception as e:
#         print(f"Error extracting keywords for {entity_name}: {e}")
#         return None


# import spacy


# def extract_entities_with_spacy(entity_name):
#     try:
#         nlp = spacy.load("en_core_web_lg")
#     except:
#         spacy.cli.download("en_core_web_lg")
#         nlp = spacy.load("en_core_web_lg")

#     page = wikipedia.page(entity_name)
#     content = page.content

#     # Process with SpaCy
#     doc = nlp(content)

#     # Extract entities by type
#     entities = {
#         "PERSON": [],
#         "ORG": [],
#         "GPE": [],  # Countries, cities
#         "DATE": [],
#         "MISC": [],
#     }

#     for ent in doc.ents:
#         if ent.label_ in entities:
#             entities[ent.label_].append(ent.text)
#         else:
#             entities["MISC"].append(ent.text)

#     # Deduplicate
#     for category in entities:
#         entities[category] = list(set(entities[category]))

#     return entities


# TODO(arnab): test
def extract_entities_with_oracle_LM(
    entity: str,
    oracle: Literal["gpt4o", "claude"] = "claude",
    other_entity: str = None,
) -> list[tuple[str, str]]:
    # system_prompt = f"""
    #     Extract key facts, relationships and attributes about {entity}.
    #     Format as a JSON with these categories:
    #     - biography: key biographical facts
    #     - achievements: major accomplishments
    #     - relationships: key people connected to the entity
    #     - organizations: affiliated organizations
    #     - places: significant locations
    #     - dates: important dates
    #     - misc: other noteworthy information
    # """
    if other_entity is None:
        system_prompt = f"""
Extrace key facts, entities, relationsships and attributes about {entity}.
Format as a JSON array, where each element is a tuple with two elements: "name of the other entity/fact" and "description of the relationship".
For example, if the entity is "Paris" the output should look like
```json
[
    ["France", "Paris is the capital of France"],
    ["Eiffel Tower", "The Eiffel Tower is located in Paris"],
    ["Louvre Museum", "The Louvre Museum is a famous museum in Paris"],
    ["City of Light", "Paris is often referred to as the City of Light"],

    ....
]
```
Make sure to include the most important and relevant facts about the entities. Give as many facts as possible.
"""

    else:
        system_prompt = f"""
Given two entities, \"{entity}\" and \"{other_entity}\", find a common link or relation between them.
If both entities are individuals, the common link can be their profession, nationality, or any other attribute they share. Their relation can be if someone is the student/teacher of the other etc.
Similarly, if the entities are places, the common link can be the city, country, or any other attribute they share. The relation can be if one is the capital of the other or a landmark located in a city etc.

Format your answer as a JSON array, where each element is a tuple with two elements: "name of the connection" and "brief explanation of how this connection is relevant to both of the entities".
For example, if the entities are "Batman" and "Ironman", the output should look like

```json
[
    ["Superheroes", "Both Batman and Ironman are iconic superheroes in the comic book world."],
    ["Gadgets", "Both characters use advanced technology and gadgets to fight crime."],
    ["Billionaires", "Both characters are wealthy individuals who use their resources to become superheroes."],
    ....
]
```
Make sure to give as many connections as possible. If you can't find any connection, just return an empty JSON array.
"""

    response = ASK_ORACLE_MODEL[oracle](system_prompt)

    # Parse the response
    try:
        lines = response.splitlines()
        if "```json" in lines[0]:
            lines = lines[1:-1]
        response = "\n".join(lines)
        response_json = json.loads(response)

    except json.JSONDecodeError:
        logger.error("Failed to parse JSON response.")
        return response

    return response_json
