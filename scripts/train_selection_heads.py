import argparse
import copy
import json
import logging
import os
import random
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin

from src.functional import (
    PatchSpec,
    free_gpu_cache,
    get_module_nnsight,
    interpret_logits,
    patch_with_baukit,
    predict_next_token,
)
from src.models import ModelandTokenizer
from src.selection.data import SelectionSample, SelectOneTask, SelectOrderTask
from src.selection.optimization import (
    get_optimal_head_mask,
    validate_q_proj_ie_on_sample_pair,
)
from src.selection.utils import KeyedSet, get_first_token_id, verify_correct_option
from src.tokens import prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import PathLike, PredictedToken, TokenizerOutput

logger = logging.getLogger(__name__)


@torch.inference_mode()
def get_counterfactual_samples_within_task(
    task: SelectOneTask | SelectOrderTask,
    mt: ModelandTokenizer,
    patch_category: str | None = None,
    clean_category: str | None = None,
    shuffle_clean_options: bool = False,
    prompt_template_idx=3,
    option_style="single_line",
    filter_by_lm_prediction: bool = True,
    distinct_options: bool = False,
    n_distractors: int = 5,
    patch_n_distractors: int | None = None,
    clean_n_distractors: int | None = None,
):
    categories = list(task.category_wise_examples.keys())
    if patch_category is None:
        patch_category = random.choice(categories)

    if patch_n_distractors is None:
        patch_n_distractors = n_distractors

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
        k=patch_n_distractors - (len(patch_must_have_options)) + 1,
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
        if clean_n_distractors is not None:
            logger.warning(
                f"Passed {clean_n_distractors=}. But distinct_options is False, so clean options will be same as patch options."
            )
        clean_options = copy.deepcopy(patch_options)
        if shuffle_clean_options:
            # Useful for the pointer experiments
            while (
                clean_options.index(clean_obj) == patch_obj_idx
                or clean_options.index(patch_type_obj) == patch_obj_idx
            ):
                random.shuffle(clean_options)
        clean_obj_idx = clean_options.index(clean_obj)

    else:
        if clean_n_distractors is None:
            clean_n_distractors = n_distractors
        other_categories = random.sample(
            list(set(categories) - {patch_category, clean_category}),
            k=clean_n_distractors - (len(clean_must_have_options)) + 1,
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
        while clean_options.index(clean_obj) == patch_obj_idx:
            random.shuffle(clean_options)
        clean_obj_idx = clean_options.index(clean_obj)

    logger.info(f"{clean_obj_idx=} | {clean_options}")

    kwargs = dict(
        prompt_template=task.prompt_templates[prompt_template_idx],
        default_option_style=option_style,
    )
    print(f"{type(task)=}")
    if isinstance(task, SelectOrderTask):
        patch_metadata = {
            "track_type_obj_idx": clean_obj_idx,
            "track_type_obj": patch_options[clean_obj_idx],
            "track_type_obj_token_id": get_first_token_id(
                patch_options[clean_obj_idx], mt.tokenizer, prefix=" "
            ),
        }
        clean_metadata = {
            "track_type_obj_idx": patch_obj_idx,
            "track_type_obj": clean_options[patch_obj_idx],
            "track_type_obj_token_id": get_first_token_id(
                clean_options[patch_obj_idx], mt.tokenizer, prefix=" "
            ),
        }
    elif isinstance(task, SelectOneTask):
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
    else:
        raise NotImplementedError(f"Unsupported task type: {type(task)}")

    patch_sample = SelectionSample(
        subj=patch_subj,
        obj=patch_obj,
        answer=patch_obj,
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
        answer=clean_obj,
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
            # sample.metadata["tokenized"] = tokenized.data
            logger.info(sample.prompt())
            logger.info(
                f"{sample.subj} | {sample.category} -> {sample.obj} | pred={[str(p) for p in predictions]}"
            )
            if not is_correct:
                logger.error(
                    f'Prediction mismatch: {track_options[list(track_options.keys())[0]]}["{mt.tokenizer.decode(predictions[0].token_id)}"] != {sample.ans_token_id}["{mt.tokenizer.decode(sample.ans_token_id)}"]'
                )
                return get_counterfactual_samples_within_task(
                    mt=mt,
                    task=task,
                    patch_category=patch_category,
                    clean_category=clean_category,
                    shuffle_clean_options=shuffle_clean_options,
                    prompt_template_idx=prompt_template_idx,
                    option_style=option_style,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    distinct_options=distinct_options,
                    n_distractors=n_distractors,
                )
            sample.prediction = predictions

    return patch_sample, clean_sample


@torch.inference_mode()
def prepare_dataset(
    mt: ModelandTokenizer,
    select_task: SelectOneTask | SelectOrderTask,
    train_limit: int = 512,
    validation_limit: int = 256,
    prompt_template_idx: int = 3,
    option_style: str = "single_line",
    distinct_options: bool = True,
):
    """
    Prepare the dataset for training and validation.
    """
    limit = train_limit + validation_limit
    dataset = []
    while len(dataset) < limit:
        patch_n_distractors = random.choice(range(1, 7))
        clean_n_distractors = random.choice(range(1, 7))
        patch_sample, clean_sample = get_counterfactual_samples_within_task(
            task=select_task,
            mt=mt,
            patch_n_distractors=patch_n_distractors,
            clean_n_distractors=clean_n_distractors,
            prompt_template_idx=prompt_template_idx,
            option_style=option_style,
            distinct_options=distinct_options,
            filter_by_lm_prediction=True,
        )
        dataset.append((clean_sample, patch_sample))
        if len(dataset) % 100 == 0:
            logger.debug("=" * 100)
            logger.info(f"Prepared {len(dataset)} / {limit} samples")
            logger.debug("=" * 100)
            free_gpu_cache()

    return dataset[:train_limit], dataset[train_limit:limit]


@torch.inference_mode()
def validate(
    mt: ModelandTokenizer,
    validation_set: list[tuple[SelectionSample, SelectionSample]],
    selected_heads: list[int],
):
    validation_results = []
    for clean_sample, patch_sample in validation_set:
        result = validate_q_proj_ie_on_sample_pair(
            mt=mt,
            clean_sample=clean_sample,
            patch_sample=patch_sample,
            heads=selected_heads,
            query_indices={-3: -3, -2: -2, -1: -1},
            verify_head_behavior_on=None,
            # amplification_scale=1.5
        )
        validation_results.append(result)
        print("=" * 80)

    before_intervention = []
    after_intervention = []

    for intervention_result in validation_results:
        clean_sample = intervention_result["clean_sample"]
        patch_sample = intervention_result["patch_sample"]

        clean_obj = clean_sample.ans_token_id
        target_obj = clean_sample.metadata["track_type_obj_token_id"]

        before_intervention.append(
            {
                "clean_rank": intervention_result["clean_track"][clean_obj][0],
                "clean_logit": intervention_result["clean_track"][clean_obj][1].logit,
                "target_rank": intervention_result["clean_track"][target_obj][0],
                "target_logit": intervention_result["clean_track"][target_obj][1].logit,
            }
        )

        after_intervention.append(
            {
                "clean_rank": intervention_result["int_track"][clean_obj][0],
                "clean_logit": intervention_result["int_track"][clean_obj][1].logit,
                "target_rank": intervention_result["int_track"][target_obj][0],
                "target_logit": intervention_result["int_track"][target_obj][1].logit,
            }
        )

    clean_rank_delta = [
        after["clean_rank"] - before["clean_rank"]
        for before, after in zip(before_intervention, after_intervention)
    ]
    target_rank_delta = [
        after["target_rank"] - before["target_rank"]
        for before, after in zip(before_intervention, after_intervention)
    ]

    clean_rank_delta, target_rank_delta = np.array(clean_rank_delta), np.array(
        target_rank_delta
    )
    print(
        f"clean_rank_delta: {clean_rank_delta.mean():.4f} ± {clean_rank_delta.std():.4f}"
    )
    print(
        f"target_rank_delta: {target_rank_delta.mean():.4f} ± {target_rank_delta.std():.4f}"
    )

    clean_rank_after_intervention = [
        after["clean_rank"] for after in after_intervention
    ]
    clean_rank_after_intervention = np.array(clean_rank_after_intervention)
    print(
        f"clean_rank_after_intervention: {clean_rank_after_intervention.mean():.4f} ± {clean_rank_after_intervention.std():.4f}"
    )

    target_rank_after_intervention = [
        after["target_rank"] for after in after_intervention
    ]
    target_rank_after_intervention = np.array(target_rank_after_intervention)
    print(
        f"target_rank_after_intervention: {target_rank_after_intervention.mean():.4f} ± {target_rank_after_intervention.std():.4f}"
    )

    clean_logit_delta = [
        after["clean_logit"] - before["clean_logit"]
        for before, after in zip(before_intervention, after_intervention)
    ]
    target_logit_delta = [
        after["target_logit"] - before["target_logit"]
        for before, after in zip(before_intervention, after_intervention)
    ]
    clean_logit_delta, target_logit_delta = np.array(clean_logit_delta), np.array(
        target_logit_delta
    )
    print(
        f"clean_logit_delta: {clean_logit_delta.mean():.4f} ± {clean_logit_delta.std():.4f}"
    )
    print(
        f"target_logit_delta: {target_logit_delta.mean():.4f} ± {target_logit_delta.std():.4f}"
    )

    clean_logit_after_intervention = [
        after["clean_logit"] for after in after_intervention
    ]
    clean_logit_after_intervention = np.array(clean_logit_after_intervention)
    print(
        f"clean_logit_after_intervention: {clean_logit_after_intervention.mean():.4f} ± {clean_logit_after_intervention.std():.4f}"
    )

    target_logit_after_intervention = [
        after["target_logit"] for after in after_intervention
    ]
    target_logit_after_intervention = np.array(target_logit_after_intervention)
    print(
        f"target_logit_after_intervention: {target_logit_after_intervention.mean():.4f} ± {target_logit_after_intervention.std():.4f}"
    )

    counter_patch_type_top_option = 0
    failed_cases = []

    for intervention_result in validation_results:
        clean_sample = intervention_result["clean_sample"]
        patch_sample = intervention_result["patch_sample"]
        int_track = intervention_result["int_track"]
        clean_track = intervention_result["clean_track"]
        if (
            int_track[list(int_track.keys())[0]][1].token_id
            == clean_sample.metadata["track_type_obj_token_id"]
        ):
            counter_patch_type_top_option += 1
        else:
            failed_cases.append(
                {
                    "clean_sample": clean_sample,
                    "patch_sample": patch_sample,
                    "int_track": int_track,
                    "clean_track": clean_track,
                }
            )

    top_1_accuracy = counter_patch_type_top_option / len(validation_results)
    logger.debug("=" * 100)
    print(
        f"Counterfactual patching accuracy: {top_1_accuracy:.4f} ({counter_patch_type_top_option}/{len(validation_results)})"
    )
    logger.debug("=" * 100)


def find_optimal_masks(
    mt: ModelandTokenizer,
    select_task: SelectOneTask | SelectOrderTask,
    save_path: PathLike,
    train_limit: int = 512,
    validation_limit: int = 256,
    prompt_template_idx: int = 3,
    n_epochs: int = 20,
    batch_size: int = 16,
    option_style: str = "single_line",
    distinct_options: bool = True,
):
    train_set, validation_set = prepare_dataset(
        mt=mt,
        select_task=select_task,
        train_limit=train_limit,
        validation_limit=validation_limit,
        prompt_template_idx=prompt_template_idx,
        option_style=option_style,
        distinct_options=distinct_options,
    )
    optimal_masks, losses = get_optimal_head_mask(
        mt=mt,
        train_set=train_set,
        learning_rate=1e-2,
        n_epochs=n_epochs,
        lamb=2e-2,
        batch_size=batch_size,
        query_indices=[-3, -2, -1],
        save_path=save_path,
        save_step=5,
        cache_q_states_before=False,  # not suitable with larger training sets
    )
    selected_heads = (
        torch.nonzero(optimal_masks > 0.5, as_tuple=False).to(dtype=torch.int).tolist()
    )
    selected_heads = [(layer_idx, head_idx) for layer_idx, head_idx in selected_heads]
    logger.info(f"Selected heads: {selected_heads}")

    validate(mt=mt, validation_set=validation_set, selected_heads=selected_heads)


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
        ],
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model identifier",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="selection/optimized_heads",
        help="Directory to save test results",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="objects",
        help="Category Type",
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
        choices=["single_line", "numbered"],
        help="Option style to use",
    )

    parser.add_argument(
        "--not_distinct",
        action="store_true",
        help="Whether to use distinct options for each sample",
    )

    parser.add_argument(
        "--train_limit",
        type=int,
        default=512,
        help="Limit the number of training samples",
    )

    parser.add_argument(
        "--validation_limit",
        type=int,
        default=256,
        help="Limit the number of validation samples",
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(f"Arguments: {args}")

    # loading the model
    mt = ModelandTokenizer(
        model_key=args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # quantization_config = BitsAndBytesConfig(
        #     # load_in_4bit=True
        #     load_in_8bit=True
        # )
        attn_implementation="eager",
    )

    # load the selection class
    select_task = SelectOneTask.load(
        path=os.path.join(
            env_utils.DEFAULT_DATA_DIR, "selection", f"{args.category}.json"
        )
    )
    select_task.filter_single_token(mt.tokenizer, prefix=" ")
    logger.info(f"{select_task=}")

    # Setup cache directory
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        args.save_dir,
        mt.name.split("/")[-1],
        "same_options",
        select_task.task_name,
    )
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Saving results to {save_dir}")

    find_optimal_masks(
        mt=mt,
        select_task=select_task,
        save_path=save_dir,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        prompt_template_idx=args.prompt_temp_idx,
        option_style=args.option_style,
        distinct_options=not args.not_distinct,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )

    logger.info("#" * 100)
    logger.info("All done!")
