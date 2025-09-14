import argparse
import copy
import json
import logging
import os
import random
from dataclasses import dataclass
from itertools import product
from typing import Literal

import torch
from dataclasses_json import DataClassJsonMixin

from src.functional import (
    PatchSpec,
    interpret_logits,
    patch_with_baukit,
)
from src.models import ModelandTokenizer
from src.selection.data import (
    SelectionSample,
    SelectOneTask,
    SelectOrderTask,
    get_counterfactual_samples_within_task,
)
from src.selection.functional import cache_q_projections
from src.selection.utils import KeyedSet, get_first_token_id, verify_correct_option
from src.tokens import prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import PathLike, PredictedToken

logger = logging.getLogger(__name__)


@dataclass
class SelectionQprojPatchResult(DataClassJsonMixin):
    patch_sample: SelectionSample
    clean_sample: SelectionSample
    interested_tokens: list[int]
    base_results: dict[int, tuple[int, PredictedToken]]
    headwise_patching_effects: dict[
        tuple[int, int], dict[int, tuple[int, PredictedToken]]
    ]
    gold_results: dict[int, tuple[int, PredictedToken]] | None = None

    def __post_init__(self):
        if "track_type_obj_token_id" in self.clean_sample.metadata:
            self.track_obj_token_id = self.clean_sample.metadata[
                "track_type_obj_token_id"
            ]
        elif "track_obj_token_id" in self.clean_sample.metadata:
            self.track_obj_token_id = self.clean_sample.metadata["track_obj_token_id"]
        else:
            raise AssertionError("Set `track_obj_token_id` in metadata of clean sample")

    def head_effect(
        self, layer_idx, head_idx, metric: Literal["prob", "logit"] = "logit"
    ):
        if isinstance(self.base_results, dict):
            low_score = getattr(self.base_results[self.track_obj_token_id][1], metric)
        else:
            low_score = getattr(self.base_results[1], metric)

        patch_score = getattr(
            self.headwise_patching_effects[(layer_idx, head_idx)][
                self.track_obj_token_id
            ][1],
            metric,
        )

        # logger.debug(f"{low_score=}, {high_score=}, {patch_score=}")
        if self.gold_results is not None:
            high_score = getattr(
                self.gold_results[self.patch_sample.ans_token_id][1], metric
            )
            indirect_effect = (patch_score - low_score) / (high_score - low_score)
        else:
            indirect_effect = patch_score - low_score
        return indirect_effect

    def delist_patching_effects(self):
        self.headwise_patching_effects = {
            f"{layer_idx}_<>_{head_idx}": effect
            for (layer_idx, head_idx), effect in self.headwise_patching_effects.items()
        }

    @staticmethod
    def load_from_json(file_path: str) -> "SelectionQprojPatchResult":
        with open(file_path, "r") as f:
            data = json.load(f)
        head_wise_patching_effects = {}
        for key, value in data["headwise_patching_effects"].items():
            layer_idx, head_idx = map(int, key.split("_<>_"))
            head_wise_patching_effects[(layer_idx, head_idx)] = value
        data["headwise_patching_effects"] = head_wise_patching_effects
        if "obj_token_id" in data["patch_sample"]:
            data["patch_sample"]["ans_token_id"] = data["patch_sample"]["obj_token_id"]
        if "obj_token_id" in data["clean_sample"]:
            data["clean_sample"]["ans_token_id"] = data["clean_sample"]["obj_token_id"]
        return SelectionQprojPatchResult.from_dict(data)


@torch.inference_mode()
def calculate_query_patching_results_for_sample_pair(
    mt: ModelandTokenizer,
    clean_sample: SelectionSample,
    patch_sample: SelectionSample,
    heads: list[tuple[int, int]],  # layer_idx, head_idx
    query_indices: list[int] = [-3, -2, -1],
) -> SelectionQprojPatchResult:
    interested_tokens = [
        patch_sample.ans_token_id,
        clean_sample.ans_token_id,
        clean_sample.metadata["track_type_obj_token_id"],
    ]

    patch_tokenized = prepare_input(prompts=patch_sample.prompt(), tokenizer=mt)

    # cache the query states + gold results with the patch sample
    # query_locations = [
    #     (layer_idx, head_idx, query_idx)
    #     for layer_idx, head_idx in heads
    #     for query_idx in query_indices
    # ]
    all_q_projections, patch_out = cache_q_projections(
        mt=mt,
        input=patch_tokenized,
        heads=heads,
        token_indices=[query_indices],
        return_output=True,
    )
    all_q_projections = all_q_projections[0]
    logger.debug(len(all_q_projections))

    patch_logits = patch_out.logits[:, -1, :].squeeze()
    patch_precitions, patch_track = interpret_logits(
        tokenizer=mt,
        logits=patch_logits,
        interested_tokens=interested_tokens,
    )
    logger.debug(f"patch_prediction={[str(pred) for pred in patch_precitions]}")
    logger.debug(f"patch_track={patch_track}")

    # run the clean sample to get base results
    clean_tokenized = prepare_input(prompts=clean_sample.prompt(), tokenizer=mt)
    clean_out = patch_with_baukit(
        mt=mt,
        inputs=clean_tokenized,
        patches=[],
    )
    base_logits = clean_out.logits[:, -1, :].squeeze()
    base_predictions, base_track = interpret_logits(
        tokenizer=mt,
        logits=base_logits,
        interested_tokens=interested_tokens,
    )
    logger.debug(f"base_prediction={[str(pred) for pred in base_predictions]}")
    logger.debug(f"base_track={base_track}")

    # patching the heads one by one
    logger.debug("patching query states for each head one by one")
    head_wise_patching_effects = {}
    counter = 0
    for layer_idx, head_idx in heads:
        q_proj_patch = [
            PatchSpec(
                location=(
                    mt.attn_module_name_format.format(layer_idx) + ".q_proj",
                    head_idx,
                    query_idx,
                ),
                patch=all_q_projections[(layer_idx, head_idx, query_idx)],
            )
            for query_idx in query_indices
        ]
        out = patch_with_baukit(
            mt=mt,
            inputs=clean_tokenized,
            patches=q_proj_patch,
        )
        logits = out.logits[:, -1, :].squeeze()
        _, track = interpret_logits(
            tokenizer=mt, logits=logits, interested_tokens=interested_tokens
        )
        head_wise_patching_effects[(layer_idx, head_idx)] = track
        counter += 1

        if counter % 128 == 0:
            logger.debug(
                f"Got patching results for {counter}/{len(heads)} ({counter/len(heads):.2%})"
            )

    return SelectionQprojPatchResult(
        patch_sample=patch_sample,
        clean_sample=clean_sample,
        interested_tokens=interested_tokens,
        base_results=base_track,
        gold_results=patch_track,
        headwise_patching_effects=head_wise_patching_effects,
    )


@torch.inference_mode()
def cache_attention_patterns_for_selection_samples(
    mt: ModelandTokenizer,
    save_dir: PathLike,
    category_type: str = "profession",
    n_distractors: int = 5,
    prompt_template_idx: int = 3,
    option_style: str = "single_line",
    query_indices: list[int] = [-1],
    limit: int = 100,
):
    save_dir = os.path.join(save_dir, category_type)
    os.makedirs(save_dir, exist_ok=True)
    select_task = SelectOneTask.load(
        path=os.path.join(
            env_utils.DEFAULT_DATA_DIR, "selection", f"{category_type}.json"
        )
    )
    all_heads = list(product(range(mt.n_layer), range(mt.config.num_attention_heads)))

    sample_idx = 0
    for sample_idx in range(limit):
        logger.info(f"Processing sample pair {sample_idx + 1}/{limit}")
        patch_sample, clean_sample = get_counterfactual_samples_within_task(
            mt=mt,
            task=select_task,
            prompt_template_idx=prompt_template_idx,
            option_style=option_style,
            n_distractors=n_distractors,
            distinct_options=True,  #! Makes sure that the value from patch sample is different from the patch type option in the clean sample
            filter_by_lm_prediction=True,
        )

        q_states_patching_results = calculate_query_patching_results_for_sample_pair(
            mt=mt,
            clean_sample=clean_sample,
            patch_sample=patch_sample,
            heads=all_heads,
            query_indices=query_indices,
        )
        q_states_patching_results.delist_patching_effects()
        if "tokenized" in q_states_patching_results.patch_sample.metadata:
            q_states_patching_results.patch_sample.metadata.pop("tokenized")
        if "tokenized" in q_states_patching_results.clean_sample.metadata:
            q_states_patching_results.clean_sample.metadata.pop("tokenized")
        file_path = os.path.join(save_dir, f"sample_pair_{sample_idx:04d}.json")
        with open(file_path, "w") as f:
            json.dump(
                q_states_patching_results.to_dict(),
                f,
                indent=4,
            )
        logger.info("=" * 80)


#! python -m test_suite.test_01_real_entities --model="meta-llama/Llama-3.3-70B-Instruct" --limit="1000"
#! append "|& tee <log_path>" to save execution logs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache selection states for language models"
    )
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "google/gemma-2-27b-it",
        ],
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model identifier",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Number of samples to generate and cache",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="selection/q_states_patching",
        help="Directory to save test results",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="objects",
        help="Category Type",
    )

    parser.add_argument(
        "--n_distractors",
        type=int,
        default=5,
        help="Number of distractors to use",
    )

    parser.add_argument(
        "--prompt_temp_idx",
        type=int,
        default=3,
        help="Prompt template index to use",
    )

    parser.add_argument(
        "--option_style",
        type=str,
        default="single_line",
        choices=["single_line", "multi_line"],
        help="Option style to use",
    )

    parser.add_argument(
        "--query_indices",
        type=list[int],
        default=[-3, -2, -1],
        help="Indices of the query token to patch (default: last 3 tokens)",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(f"Arguments: {args}")

    # loading the model
    mt = ModelandTokenizer(
        model_key=args.model,
        torch_dtype=torch.bfloat16,
        # device_map=device_map,
        device_map="auto",
        # quantization_config = BitsAndBytesConfig(
        #     # load_in_4bit=True
        #     load_in_8bit=True
        # )
        attn_implementation="eager",
    )

    # Setup cache directory
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        args.save_dir,
        mt.name.split("/")[-1],
    )
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Saving results to {save_dir}")
    cache_attention_patterns_for_selection_samples(
        mt=mt,
        save_dir=save_dir,
        category_type=args.category,
        n_distractors=args.n_distractors,
        prompt_template_idx=args.prompt_temp_idx,
        option_style=args.option_style,
        query_indices=args.query_indices,
        limit=args.limit,
    )
    logger.info("#" * 100)
    logger.info("All done!")
