import argparse
import logging
import os
import json

import numpy as np
import torch
import transformers
import itertools
import random
from tqdm import tqdm
from src.utils.typing import TokenizerOutput

from src.functional import free_gpu_cache, get_tick_marker, get_hs, detensorize
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils
from src.probing.utils import (
    ProbingPrompt,
    ProbingLatents,
    prepare_probing_input,
    get_lm_generated_answer,
    check_if_answer_is_correct,
)
import hashlib

logger = logging.getLogger(__name__)


logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")

###########################################################################
INSTRUCTION = f"""Given two entities, find a common link or relation between them.
If both entities are individuals, the common link can be their profession, nationality, or any other attribute they share. Their relation can be if someone is the student/teacher of the other etc.
Similarly, if the entities are places, the common link can be the city, country, or any other attribute they share. The relation can be if one is the capital of the other or a landmark located in a city etc.
If there is no connection just answer "None"."""

# Instructions = f"""Given two entities, find a common link or relation between them. If there is no connection just answer "None"."""

BLOCK_SEPARATOR = "\n#"
QUESTION_MARKER = "\nQ: "
ANSWER_MARKER = "\nA:"

ICL_EXAMPLES = """#
Captain America and Deathstroke
A: They are both comic book characters and enhanced super soldiers.
#
Q: Tiger Woods and Phil Mickelson
A: They are both professional golfers.
#
Q: Rome and Italy
A: Rome is the capital city of Italy.
#
Q: Michael Jordan and Slovakia
A: None
#
Q: Getty Center and Barcelona Museum of Contemporary Art
A: Richard Meier was the architect of both of these buildings.
"""

PREFIX = f"""{INSTRUCTION}\n{ICL_EXAMPLES}"""
###########################################################################


@torch.inference_mode()
def cache_probing_latents(
    model_name: str,
    probe_class: str,
    limit: int = 10000,
    save_dir: str = "probing_latents",
):
    probe_file = os.path.join(
        env_utils.DEFAULT_DATA_DIR,
        "probe",
        probe_class if probe_class.endswith(".json") else f"{probe_class}.json",
    )
    probe_data = json.load(open(probe_file, "r"))
    class_name = probe_data["class"]
    keywords = probe_data["keywords"]

    mt = ModelandTokenizer(
        model_key=model_name,
        torch_dtype=torch.float16,
    )

    cache_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        save_dir,
        mt.name.split("/")[-1],
        class_name,
    )
    os.makedirs(cache_dir, exist_ok=True)

    entity_pairs = list(itertools.combinations(probe_data["entities"], 2))
    random.shuffle(entity_pairs)
    limit = min(limit, len(entity_pairs))

    logger.info(f"loaded probe data | {class_name=} | {len(entity_pairs)=}")

    layer_names = mt.layer_names

    save_count = 0

    for idx, entity_pair in tqdm(enumerate(entity_pairs)):
        logger.info(f"{entity_pair=}")
        prompt = prepare_probing_input(
            mt=mt,
            entities=entity_pair,
            prefix=PREFIX,
            answer_marker=ANSWER_MARKER,
            question_marker=QUESTION_MARKER,
            block_separator=BLOCK_SEPARATOR,
        )

        lm_answer = get_lm_generated_answer(
            mt=mt, prompt=prompt, block_separator=BLOCK_SEPARATOR
        )

        if lm_answer.startswith("None"):
            is_correct = False
        else:
            is_correct = check_if_answer_is_correct(
                answer=lm_answer,
                keywords=keywords,
                oracle_model="claude",
                entities=entity_pair,
            )

        logger.info(f"({get_tick_marker(is_correct)}) {lm_answer=}")

        if is_correct == False:
            logger.error(
                f"Incorrect Answer: not storing latents {entity_pair} | {lm_answer=}"
            )
            continue

        token_indices = set(
            set(range(*prompt.query_range))
            .union(set(range(*prompt.entity_ranges[0])))
            .union(set(range(*prompt.entity_ranges[1])))
        )
        token_indices = list(range(min(token_indices), max(token_indices) + 1))

        locations = list(itertools.product(layer_names, token_indices))

        latents = get_hs(
            mt=mt,
            input=TokenizerOutput(prompt.tokenized),
            locations=locations,
            return_dict=True,
        )

        probing_latents = ProbingLatents(
            prompt=prompt,
            latents=latents,
            lm_answer=lm_answer,
        )

        dnpz = detensorize(probing_latents, to_numpy=True)
        file_name = hashlib.md5(f"{entity_pair}".encode()).hexdigest()
        np.savez_compressed(
            os.path.join(cache_dir, f"{file_name}.npz"),
            **dnpz.get_dict(),
            allow_pickle=True,
        )
        save_count += 1
        logger.info(
            f"saved {save_count}/{limit} latents | lm_accuracy={save_count/(idx+1) : .3f} ({save_count}/{idx+1})"
        )
        if save_count >= limit:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
        default="meta-llama/Llama-3.2-3B",
    )

    parser.add_argument(
        "--probe_class",
        type=str,
        default="athletes/basketball.json",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="probing_latents",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(args)

    cache_probing_latents(
        model_name=args.model,
        probe_class=args.probe_class,
        limit=args.limit,
        save_dir=args.save_dir,
    )
