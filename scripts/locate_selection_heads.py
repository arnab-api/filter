import argparse
import json
import logging
import os
import random
from typing import Literal

import numpy as np
import torch

from src.functional import free_gpu_cache
from src.models import ModelandTokenizer
from src.selection.data import (
    CounterFactualSamplePair,
    CountingSample,
    CountingTask,
    MCQify_sample,
    SelectFirstTask,
    SelectionSample,
    SelectLastTask,
    SelectOneTask,
    SelectOrderTask,
    YesNoSample,
    YesNoTask,
    get_counterfactual_samples_interface,
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
    select_task: (
        SelectOneTask | CountingTask | YesNoTask | SelectFirstTask | SelectLastTask
    ),
    option_config: Literal["distinct", "same", "position"],
    save_path: PathLike,
    train_limit: int = 512,
    validation_limit: int = 256,
    prompt_template_idx: int = 3,
    option_style: str = "single_line",
    distinct_options: bool = True,
    mcqify: bool = False,
):
    """
    Prepare the dataset for training and validation.
    """
    counterfactual_sampler = get_counterfactual_samples_interface[select_task.task_name]
    limit = train_limit + validation_limit
    dataset = []
    while len(dataset) < limit:
        logger.debug(f"sample {len(dataset)+1} / {limit}")
        kwargs = {}
        if isinstance(select_task, CountingTask):
            kwargs["clean_n_options"] = random.choice(range(4, 7))
            kwargs["patch_n_options"] = random.choice(range(4, 7))
            kwargs["distinct_options"] = distinct_options
        elif isinstance(select_task, YesNoTask):
            kwargs["clean_n_options"] = random.choice(range(3, 6))
            kwargs["patch_n_options"] = random.choice(range(3, 6))
            # No distinct options for yes/no task
        elif isinstance(select_task, SelectFirstTask | SelectLastTask):
            #! this has to come before SelectOneTask since SelectFirstTask is a subclass of SelectOneTask
            kwargs["distinct_options"] = distinct_options
        elif isinstance(select_task, SelectOneTask):
            kwargs["distinct_options"] = distinct_options
            kwargs["mcqify"] = mcqify and option_config == "position"
            if option_config == "position":
                n_distractors = random.choice(range(3, 7))
                kwargs["patch_n_distractors"] = n_distractors
                kwargs["clean_n_distractors"] = n_distractors
            else:
                kwargs["patch_n_distractors"] = random.choice(range(2, 7))
                kwargs["clean_n_distractors"] = random.choice(range(2, 7))
        else:
            raise ValueError(f"Unknown task type: {type(select_task)}")

        if prompt_template_idx == -1:
            kwargs["patch_prompt_template_idx"] = random.choice(
                range(len(select_task.prompt_templates))
            )
            kwargs["clean_prompt_template_idx"] = random.choice(
                range(len(select_task.prompt_templates))
            )
        else:
            kwargs["patch_prompt_template_idx"] = prompt_template_idx
            kwargs["clean_prompt_template_idx"] = prompt_template_idx

        patch_sample, clean_sample = counterfactual_sampler(
            task=select_task,
            mt=mt,
            option_style=option_style,
            prompt_template_idx=prompt_template_idx,
            filter_by_lm_prediction=True,
            **kwargs,
        )

        if option_config == "position":
            if not mcqify:
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
            else:
                patch_sample = MCQify_sample(
                    sample=patch_sample, tokenizer=mt, start_from="a"
                )
                patch_sample.ans_token_id = get_first_token_id(
                    chr(
                        patch_sample.obj_idx + ord(patch_sample.option_label_start_from)
                    ),
                    tokenizer=mt.tokenizer,
                    prefix=" ",
                )
                clean_sample = MCQify_sample(
                    sample=clean_sample, tokenizer=mt, start_from="p"
                )
                clean_sample.ans_token_id = get_first_token_id(
                    chr(
                        clean_sample.obj_idx + ord(clean_sample.option_label_start_from)
                    ),
                    tokenizer=mt.tokenizer,
                    prefix=" ",
                )
                # pred_obj_idx = clean_sample.metadata["track_type_obj_idx"] #! to match value
                pred_obj_idx = patch_sample.obj_idx  #! to match position
                clean_sample.metadata["track_type_obj_token_id"] = get_first_token_id(
                    name=chr(ord(clean_sample.option_label_start_from) + pred_obj_idx),
                    tokenizer=mt.tokenizer,
                    prefix=" ",
                )

            logger.debug(f"Clean sample metadata: {clean_sample.metadata}")
            #! not really using patch_sample.metadata
            patch_sample.metadata = {}

        dataset.append((clean_sample, patch_sample))
        if len(dataset) % 128 == 0:
            logger.debug("=" * 100)
            logger.info(f"Prepared {len(dataset)} / {limit} samples")
            logger.debug("=" * 100)
            free_gpu_cache()

    train_set, validation_set = dataset[:train_limit], dataset[train_limit:limit]

    sample_save_dir = os.path.join(save_path, "samples")
    os.makedirs(sample_save_dir, exist_ok=True)
    train_save_path = os.path.join(sample_save_dir, "train")
    validation_save_path = os.path.join(sample_save_dir, "validation")

    for dataset, save_path in [
        (train_set, train_save_path),
        (validation_set, validation_save_path),
    ]:
        os.makedirs(save_path, exist_ok=True)
        for sample_idx, (clean_sample, patch_sample) in enumerate(dataset):
            cf_pair = CounterFactualSamplePair(
                clean_sample=clean_sample, patch_sample=patch_sample
            )
            cf_pair.detensorize()
            with open(
                os.path.join(save_path, f"sample_{sample_idx:05d}.json"),
                "w",
            ) as f:
                f.write(json.dumps(cf_pair.to_dict(), indent=2))

        logger.info(f"Saved {len(dataset)} samples to {save_path}")

    return train_set, validation_set


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


def load_dataset(
    path: PathLike, limit: int, prefix=""
) -> list[SelectionSample, SelectionSample]:
    sample_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")
    ]
    logger.info(f"Found {len(sample_files)} sample files")

    # prefix = "Recall the nationality of these people:\n"
    # prefix = "Recall which country these landmarks are located in:\n"
    # prefix = "Think about how these words sound when you say them aloud:\n"

    random.shuffle(sample_files)
    sample_files = sample_files[:limit]
    dataset = []
    for sample_file in sample_files:
        with open(sample_file, "r") as f:
            cf_pair_data = json.load(f)
        cf_pair = CounterFactualSamplePair.from_dict(cf_pair_data)
        # cf_pair.patch_sample.default_option_style = "bulleted"
        # cf_pair.clean_sample.default_option_style = "bulleted"

        cf_pair.clean_sample.prompt_template = (
            prefix + cf_pair.clean_sample.prompt_template
        )
        cf_pair.patch_sample.prompt_template = (
            prefix + cf_pair.patch_sample.prompt_template
        )
        dataset.append((cf_pair.clean_sample, cf_pair.patch_sample))

    return dataset


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
    mcqify: bool = False,
):
    indices_kwargs = {"query_indices": [-2, -1]}
    if optimization_function == get_optimal_head_mask_optimized:
        indices_kwargs["add_ques_pos_to_query_indices"] = True
    elif optimization_function == get_optimal_head_mask_prev:
        # indices_kwargs["query_indices"] = [-3, -2, -1]
        indices_kwargs["query_indices"] = [
            -1
        ]  #! faster and getting better results with only the last token

    optimal_masks, losses = optimization_function(
        mt=mt,
        train_set=train_set,
        learning_rate=1e-2,
        n_epochs=n_epochs,
        lamb=2e-2,  #! optimized for llama-70b
        batch_size=batch_size,
        save_path=save_path,
        save_step=5,
        **indices_kwargs,
    )
    selected_heads = (
        torch.nonzero(optimal_masks > 0.5, as_tuple=False).to(dtype=torch.int).tolist()
    )
    selected_heads = [(layer_idx, head_idx) for layer_idx, head_idx in selected_heads]
    logger.info(f"Selected heads ({len(selected_heads)}): {selected_heads}")

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
        default="distinct",  #! we almost always want distinct options
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

    parser.add_argument(
        "--task",
        type=str,
        choices=["select_one", "counting", "yes_no", "select_first", "select_last"],
        default="select_one",
        help="Which task to optimize",
    )

    # to mcqify the samples (for select_one task)
    parser.add_argument(
        "--mcqify",
        action="store_true",
        help="Whether to convert the samples to multiple-choice questions",
    )

    parser.add_argument(
        "--load_dataset_from",
        type=str,
        default=None,
        help="Path to load pre-saved dataset from",
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

    TASK_NAME_TO_CLASS = {
        "select_one": SelectOneTask,
        "counting": CountingTask,
        "yes_no": YesNoTask,
        "select_first": SelectFirstTask,
        "select_last": SelectLastTask,
    }

    # load the selection class
    select_task = TASK_NAME_TO_CLASS[args.task].load(
        path=os.path.join(
            env_utils.DEFAULT_DATA_DIR, "selection", f"{args.category}.json"
        )
    )
    # select_task.filter_single_token(mt.tokenizer, prefix=" ")
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
        select_task.task_name + ("_mcq" if args.mcqify else ""),
    )
    if args.opt_interface == "legacy":
        save_dir = os.path.join(save_dir, "legacy")
    logger.info(f"{save_dir=}")
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Saving results to {save_dir}")

    if args.load_dataset_from is not None:
        train_path = os.path.join(args.load_dataset_from, "train")
        validation_path = os.path.join(args.load_dataset_from, "validation")

        train_set = load_dataset(path=train_path, limit=args.train_limit)
        validation_set = load_dataset(path=validation_path, limit=args.validation_limit)
        logger.info(
            f"Loaded dataset from {args.load_dataset_from} >> {len(train_set)=}, {len(validation_set)=}"
        )

    else:
        train_set, validation_set = prepare_dataset(
            mt=mt,
            select_task=select_task,
            option_config=args.option_config,
            mcqify=args.mcqify,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            prompt_template_idx=args.prompt_temp_idx,
            option_style=args.option_style,
            distinct_options=args.option_config in ["distinct", "position"],
            save_path=save_dir,
        )

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
        mcqify=args.mcqify,
    )

    logger.info("#" * 100)
    logger.info("All done!")
