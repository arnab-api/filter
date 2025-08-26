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
    get_module_nnsight,
    interpret_logits,
    patch_with_baukit,
    predict_next_token,
)
from src.models import ModelandTokenizer
from src.selection.data import SelectionSample, SelectOneTask
from src.selection.functional import cache_q_projections
from src.selection.utils import KeyedSet, get_first_token_id, verify_correct_option
from src.tokens import prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import PathLike, PredictedToken, TokenizerOutput

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


def get_counterfactual_samples_within_task(
    task: SelectOneTask,
    patch_category: str | None = None,
    clean_category: str | None = None,
    shuffle_clean_options: bool = False,
    prompt_template_idx=3,
    option_style="single_line",
    filter_by_lm_prediction: bool = True,
    distinct_options: bool = False,
    n_distractors: int = 5,
):
    categories = list(task.category_wise_examples.keys())
    if patch_category is None:
        patch_category = random.choice(categories)

    patch_subj, patch_obj = random.sample(
        task.category_wise_examples[patch_category], 2
    )
    logger.info(
        f"Patch category: {patch_category}, subject: {patch_subj}, object: {patch_obj}"
    )

    if clean_category is None:
        clean_category = random.choice(list(set(categories) - {patch_category}))

    clean_options = task.category_wise_examples[clean_category]
    random.shuffle(clean_options)

    clean_subj, clean_obj = random.sample(
        (
            KeyedSet(clean_options, mt.tokenizer) - KeyedSet([patch_obj], mt.tokenizer)
        ).values,
        2,
    )
    logger.info(
        f"Clean category: {clean_category}, subject: {clean_subj}, object: {clean_obj}"
    )

    if distinct_options is False:
        # patch_must_have_options = [patch_obj, clean_obj]
        # clean_must_have_options = [clean_obj, patch_obj]
        patch_type_obj = patch_obj
        clean_type_obj = clean_obj
    else:
        patch_type_obj = random.choice(
            (
                KeyedSet(task.category_wise_examples[patch_category], mt.tokenizer)
                - KeyedSet([patch_obj], mt.tokenizer)
            ).values
        )
        clean_type_obj = random.choice(
            (
                KeyedSet(task.category_wise_examples[clean_category], mt.tokenizer)
                - KeyedSet([clean_obj], mt.tokenizer)
            ).values
        )

    patch_must_have_options = [patch_obj, clean_type_obj]
    clean_must_have_options = [clean_obj, patch_type_obj]

    logger.info(f"{patch_must_have_options=}")
    logger.info(f"{clean_must_have_options=}")
    logger.info(f"{clean_type_obj=}")
    logger.info(f"{patch_type_obj=}")

    patch_distractors = []
    other_categories = random.sample(
        list(set(categories) - {patch_category, clean_category}),
        k=n_distractors - (len(patch_must_have_options)) + 1,
    )

    for other_category in other_categories:
        other_examples = task.category_wise_examples[other_category]
        random.shuffle(other_examples)
        other_examples = KeyedSet(other_examples, mt.tokenizer)
        patch_distractors.append(
            random.choice(
                (
                    other_examples
                    - KeyedSet(
                        patch_must_have_options + patch_distractors,
                        tokenizer=mt.tokenizer,
                    )
                ).values
            )
        )

    patch_options = patch_must_have_options + patch_distractors
    random.shuffle(patch_options)
    patch_obj_idx = patch_options.index(patch_obj)
    logger.info(f"{patch_obj_idx=} | {patch_options}")

    if distinct_options is not True:
        clean_options = copy.deepcopy(patch_options)
        if shuffle_clean_options:
            # Useful for the pointer experiments
            while (
                clean_options.index(clean_obj) == patch_obj_idx
                or clean_options.index(patch_obj) == patch_obj_idx
            ):
                random.shuffle(clean_options)
        clean_obj_idx = clean_options.index(clean_obj)

    else:
        other_categories = random.sample(
            list(set(categories) - {patch_category, clean_category}),
            k=n_distractors - (len(clean_must_have_options)) + 1,
        )
        clean_distractors = []
        for other_category in other_categories:
            other_examples = task.category_wise_examples[other_category]
            random.shuffle(other_examples)
            other_examples = KeyedSet(other_examples, mt.tokenizer)
            clean_distractors.append(
                random.choice(
                    (
                        other_examples
                        - KeyedSet(
                            clean_must_have_options + clean_distractors,
                            tokenizer=mt.tokenizer,
                        )
                    ).values
                )
            )
        clean_options = clean_must_have_options + clean_distractors
        random.shuffle(clean_options)
        clean_obj_idx = clean_options.index(clean_obj)

    logger.info(f"{clean_obj_idx=} | {clean_options}")

    kwargs = dict(
        prompt_template=task.prompt_templates[prompt_template_idx],
        default_option_style=option_style,
    )
    patch_metadata = {
        "track_category": clean_category,
        "track_type_obj": clean_type_obj,
        "track_type_obj_idx": patch_options.index(clean_type_obj),
        "track_type_obj_token_id": get_first_token_id(
            clean_type_obj, mt.tokenizer, prefix=" "
        ),
    }
    clean_metadata = {
        "track_category": patch_category,
        "track_type_obj": patch_type_obj,
        "track_type_obj_idx": clean_options.index(patch_type_obj),
        "track_type_obj_token_id": get_first_token_id(
            patch_type_obj, mt.tokenizer, prefix=" "
        ),
    }

    patch_sample = SelectionSample(
        subj=patch_subj,
        obj=patch_obj,
        obj_idx=patch_obj_idx,
        ans_token_id=get_first_token_id(patch_obj, mt.tokenizer, prefix=" "),
        options=patch_options,
        category=patch_category,
        metadata=patch_metadata,
        **kwargs,
    )
    clean_sample = SelectionSample(
        subj=clean_subj,
        obj=clean_obj,
        obj_idx=clean_obj_idx,
        ans_token_id=get_first_token_id(clean_obj, mt.tokenizer, prefix=" "),
        options=clean_options,
        category=clean_category,
        metadata=clean_metadata,
        **kwargs,
    )

    if filter_by_lm_prediction:
        test_samples = [patch_sample, clean_sample]
        if distinct_options is True:
            clean_sample_2 = copy.deepcopy(patch_sample)
            clean_sample_2.options = clean_options
            clean_sample_2.obj = clean_sample.metadata["track_type_obj"]
            clean_sample_2.obj_idx = clean_sample.metadata["track_type_obj_idx"]
            clean_sample_2.ans_token_id = clean_sample.metadata[
                "track_type_obj_token_id"
            ]
            test_samples.append(clean_sample_2)

        for sample in test_samples:
            tokenized = prepare_input(tokenizer=mt, prompts=sample.prompt())
            is_correct, predictions, track_options = verify_correct_option(
                mt=mt, target=sample.obj, options=sample.options, input=tokenized
            )
            sample.metadata["tokenized"] = tokenized.data
            logger.info(sample.prompt())
            logger.info(
                f"{sample.subj} | {sample.category} -> {sample.obj} | pred={[str(p) for p in predictions]}"
            )
            if not is_correct:
                logger.error(
                    f'Prediction mismatch: {track_options[list(track_options.keys())[0]]}["{mt.tokenizer.decode(predictions[0].token_id)}"] != {sample.ans_token_id}["{mt.tokenizer.decode(sample.ans_token_id)}"]'
                )
                return get_counterfactual_samples_within_task(
                    task=task,
                    patch_category=patch_category,
                    clean_category=clean_category,
                    shuffle_clean_options=shuffle_clean_options,
                    prompt_template_idx=prompt_template_idx,
                    option_style=option_style,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                )
            sample.prediction = predictions

    return patch_sample, clean_sample


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
    query_locations = [
        (layer_idx, head_idx, query_idx)
        for layer_idx, head_idx in heads
        for query_idx in query_indices
    ]
    all_q_projections, patch_out = cache_q_projections(
        mt=mt,
        input=patch_tokenized,
        query_locations=query_locations,
        return_output=True,
    )
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

        if counter % 100 == 0:
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
        q_states_patching_results.patch_sample.metadata.pop("tokenized")
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
        default="profession",
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
