import argparse
import logging
import os
import random
from typing import Literal

import numpy as np
import torch

from src.functional import free_gpu_cache
from src.models import ModelandTokenizer
from src.selection.data import (
    SelectionSample,
    SelectOneTask,
    SelectOrderTask,
    get_counterfactual_samples_within_task,
)
from src.selection.optimization import (
    get_optimal_head_mask_optimized,
    get_optimal_head_mask_prev,
    validate_q_proj_ie_on_sample_pair,
)
from src.selection.utils import get_first_token_id
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)

optimization_interface = {
    "legacy": get_optimal_head_mask_prev,
    "updated": get_optimal_head_mask_optimized,
}


@torch.inference_mode()
def prepare_dataset(
    mt: ModelandTokenizer,
    select_task: SelectOneTask | SelectOrderTask,
    option_config: Literal["distinct", "same", "position"],
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
        logger.debug(f"sample {len(dataset)+1} / {limit}")
        if option_config == "position":
            n_distractors = random.choice(range(3, 7))
            patch_n_distractors = n_distractors
            clean_n_distractors = n_distractors
        else:
            patch_n_distractors = random.choice(range(2, 7))
            clean_n_distractors = random.choice(range(2, 7))
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
        if option_config == "position":
            clean_sample.metadata = {
                "track_category": "position",
                "track_type_obj_idx": patch_sample.obj_idx,
                "track_type_obj": clean_sample.options[patch_sample.obj_idx],
                "track_type_obj_token_id": get_first_token_id(
                    name=clean_sample.options[patch_sample.obj_idx],
                    tokenizer=mt.tokenizer,
                    prefix=" ",
                ),
            }
            logger.debug(f"Clean sample metadata: {clean_sample.metadata}")
            #! not really using patch_sample.metadata
            patch_sample.metadata = {}
        dataset.append((clean_sample, patch_sample))
        if len(dataset) % 128 == 0:
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
            query_indices={-2: -2, -1: -1},
            add_ques_pos_to_query_indices=True,
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
    option_config: Literal["distinct", "same", "position"],
    save_path: PathLike,
    train_limit: int = 512,
    validation_limit: int = 256,
    prompt_template_idx: int = 3,
    n_epochs: int = 20,
    batch_size: int = 16,
    option_style: str = "single_line",
    distinct_options: bool = True,
    optimization_function=get_optimal_head_mask_optimized,
):
    train_set, validation_set = prepare_dataset(
        mt=mt,
        select_task=select_task,
        option_config=option_config,
        train_limit=train_limit,
        validation_limit=validation_limit,
        prompt_template_idx=prompt_template_idx,
        option_style=option_style,
        distinct_options=distinct_options,
    )
    indices_kwargs = {"query_indices": [-2, -1]}
    if optimization_function == get_optimal_head_mask_optimized:
        indices_kwargs["add_ques_pos_to_query_indices"] = True
    elif optimization_function == get_optimal_head_mask_prev:
        indices_kwargs["query_indices"] = [-3, -2, -1]

    optimal_masks, losses = optimization_function(
        mt=mt,
        train_set=train_set,
        learning_rate=1e-2,
        n_epochs=n_epochs,
        lamb=2e-2,
        batch_size=batch_size,
        save_path=save_path,
        save_step=5,
        **indices_kwargs,
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
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "google/gemma-2-27b-it",
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
        # default=-1,
        default=3,
        help="Prompt template index to use (-1 for random selection from available templates)",
    )

    parser.add_argument(
        "--option_style",
        type=str,
        default="single_line",
        choices=["single_line", "numbered"],
        help="Option style to use",
    )

    parser.add_argument(
        "--option_config",
        choices=["distinct", "same", "position"],
        help="Configuration for option selection",
        default="distinct",
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

    parser.add_argument(
        "--opt_interface",
        type=str,
        choices=["legacy", "updated"],
        default="legacy",  # ! when question comes after (most of the cases) "legacy" will be much (6x) faster
        help="Which optimization interface to use",
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
    opt_config_dir = {
        "distinct": "distinct_options",
        "same": "same_options",
        "position": "ans_pointer",
    }
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        args.save_dir,
        mt.name.split("/")[-1],
        opt_config_dir[args.option_config],
        select_task.task_name,
    )
    if args.opt_interface == "legacy":
        save_dir = os.path.join(save_dir, "legacy")
    logger.info(f"{save_dir=}")
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
        distinct_options=args.option_config in ["distinct", "position"],
        option_config=args.option_config,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        optimization_function=optimization_interface[args.opt_interface],
    )

    logger.info("#" * 100)
    logger.info("All done!")
