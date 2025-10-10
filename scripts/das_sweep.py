import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin

from src.models import ModelandTokenizer
from src.selection.data import (
    CounterFactualSamplePair,
    CountingSample,
    SelectionSample,
    YesNoSample,
)
from src.selection.optimization import (
    get_optimal_rotation,
    validate_projections_on_sample_pair,
)
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import PathLike, PredictedToken

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult_on_sample_pair(DataClassJsonMixin):
    source: Union[SelectionSample, CountingSample, YesNoSample]
    destination: Union[SelectionSample, CountingSample, YesNoSample]
    patched_predictions: list[PredictedToken]
    patched_track: dict[int, tuple[int, PredictedToken]]
    clean_predictions: list[PredictedToken]
    clean_track: dict[int, tuple[int, PredictedToken]]
    track_tokens: dict[str, int] | None = None


@dataclass
class ValidationResults(DataClassJsonMixin):
    layer: str
    projection_dim: int | Literal["full"]
    results: list[ValidationResult_on_sample_pair]
    summary: dict[str, float]


def das_sweep(
    mt: ModelandTokenizer,
    train_set: list[SelectionSample, SelectionSample],
    validation_set: list[SelectionSample, SelectionSample],
    layers: list[str],
    token_mapping: dict[int, int],
    projection_dim: int,
    epochs: int,
    learning_rate: float,
    ortho_reg: float,
    save_dir: PathLike,
    projection_path: PathLike = None,
    batch_size: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)
    for layer in layers:
        logger.info("#" * 100)
        logger.info(f"Processing layer: {layer}")

        layer_save_dir = os.path.join(save_dir, layer)
        os.makedirs(layer_save_dir, exist_ok=True)

        if projection_path is None:
            # train optimal rotator
            optimal_rotator, training_losses = get_optimal_rotation(
                mt=mt,
                train_set=train_set,
                layers=[layer],
                token_mapping=token_mapping,
                rotation_n_dim=projection_dim,
                learning_rate=learning_rate,
                ortho_reg=ortho_reg,
                n_epochs=epochs,
                batch_size=batch_size,
                save_path=layer_save_dir,
            )
            # save the losses
            with open(os.path.join(layer_save_dir, "training_losses.json"), "w") as f:
                json.dump(training_losses, f, indent=4)

        else:
            # load the optimal rotator
            layer_proj_path = os.path.join(projection_path, layer)
            assert os.path.exists(
                layer_proj_path
            ), f"Projection path {layer_proj_path} does not exist"
            epoch_files = [
                f
                for f in os.listdir(layer_proj_path)
                if f.startswith("epoch_") and f.endswith(".pt")
            ]
            epoch_files = sorted(
                epoch_files,
                key=lambda x: int(x.split("_")[1].split(".")[0]),
                reverse=True,
            )
            assert len(epoch_files) > 0, f"No epoch files found in {layer_proj_path}"
            final_epoch_file = os.path.join(layer_proj_path, epoch_files[0])
            logger.info(f"Loading projection from {final_epoch_file}")
            optimal_rotator = torch.load(final_epoch_file, weights_only=False)
            assert layer in optimal_rotator, f"Layer {layer} not found in projection"

        # validate the learned rotator

        results = []
        for destination, source in validation_set:
            # destination = copy.deepcopy(destination)
            # source = copy.deepcopy(source)
            # destination.default_option_style = "bulleted"
            # source.prompt_template = select_task.prompt_templates[2]
            # destination.prompt_template = select_task.prompt_templates[2]

            # source_tokenized = prepare_input(prompts=[source.prompt()], tokenizer=mt)
            # destination_tokenized = prepare_input(prompts=[destination.prompt()], tokenizer=mt)

            track_tokens = {
                "clean_obj": destination.ans_token_id,
                "corrupt_obj": source.ans_token_id,
                "target_obj": destination.metadata["track_type_obj_token_id"],
            }

            pair_result = validate_projections_on_sample_pair(
                mt=mt,
                destination_sample=destination,
                source_sample=source,
                rotators=optimal_rotator,
                rotate_dimensions=projection_dim,
                token_mapping=token_mapping,
                must_track_tokens=list(track_tokens.values()),
                debug=True,
                return_clean_predictions=True,
            )
            das_patched_pred = pair_result["patched_predictions"]
            das_patched_track = pair_result["patched_track"]
            destination_pred = pair_result["destination_predictions"]
            destination_track = pair_result["destination_track"]

            results.append(
                ValidationResult_on_sample_pair(
                    source=source,
                    destination=destination,
                    patched_predictions=das_patched_pred,
                    patched_track=das_patched_track,
                    clean_predictions=destination_pred,
                    clean_track=destination_track,
                    track_tokens=track_tokens,
                )
            )

        # summarize the validation results
        track_token_types = ["clean_obj", "target_obj"]
        summary = {}
        for token_type in track_token_types:
            print(f"{token_type}")
            ranks = {"clean": [], "patch": []}
            logits = {"clean": [], "patch": []}
            for result in results:
                target_tok_id = result.track_tokens[token_type]
                clean_rank, clean_pred = result.clean_track[target_tok_id]
                patch_rank, patch_pred = result.patched_track[target_tok_id]
                ranks["clean"].append(clean_rank)
                ranks["patch"].append(patch_rank)
                logits["clean"].append(clean_pred.logit)
                logits["patch"].append(patch_pred.logit)

            attr = {
                "rank": ranks,
                "logit": logits,
            }
            for key in attr:
                clean = np.array(attr[key]["clean"])
                patch = np.array(attr[key]["patch"])
                delta = patch - clean
                print(
                    f"{key}: clean {clean.mean():.2f} ± {clean.std():.2f} -> patch {patch.mean():.2f} ± {patch.std():.2f} | delta {delta.mean():.2f} ± {delta.std():.2f}"
                )

            summary[token_type] = attr

        top_1_accuracy = 0
        for result in results:
            target_tok_id = result.destination.metadata["track_type_obj_token_id"]
            patched_track = result.patched_track
            if (
                patched_track[list(patched_track.keys())[0]][1].token_id
                == target_tok_id
            ):
                top_1_accuracy += 1

        top_1_accuracy = top_1_accuracy / len(results)
        summary["top_1_accuracy"] = top_1_accuracy
        print(f"Top-1 accuracy: {top_1_accuracy:.3f}")

        validation_results = ValidationResults(
            layer=layer,
            projection_dim=projection_dim,
            results=results,
            summary=summary,
        )
        with open(os.path.join(layer_save_dir, "validation_results.json"), "w") as f:
            f.write(validation_results.to_json(indent=4))
        logger.info("#" * 100)


def residual_patching_sweep(
    mt: ModelandTokenizer,
    validation_set: list[SelectionSample, SelectionSample],
    layers: list[str],
    token_mapping: dict[int, int],
    save_dir: PathLike,
):
    os.makedirs(save_dir, exist_ok=True)
    for layer in layers:
        logger.info("#" * 100)
        logger.info(f"Processing layer: {layer}")
        layer_save_dir = os.path.join(save_dir, layer)
        os.makedirs(layer_save_dir, exist_ok=True)

        results = []
        for destination, source in validation_set:
            # destination = copy.deepcopy(destination)
            # source = copy.deepcopy(source)
            # destination.default_option_style = "bulleted"
            # source.prompt_template = select_task.prompt_templates[2]
            # destination.prompt_template = select_task.prompt_templates[2]

            # source_tokenized = prepare_input(prompts=[source.prompt()], tokenizer=mt)
            # destination_tokenized = prepare_input(prompts=[destination.prompt()], tokenizer=mt)

            track_tokens = {
                "clean_obj": destination.ans_token_id,
                "corrupt_obj": source.ans_token_id,
                "target_obj": destination.metadata["track_type_obj_token_id"],
            }

            pair_result = validate_projections_on_sample_pair(
                mt=mt,
                destination_sample=destination,
                source_sample=source,
                rotators={layer: None},
                rotate_dimensions="full",
                token_mapping=token_mapping,
                must_track_tokens=list(track_tokens.values()),
                debug=True,
                return_clean_predictions=True,
            )
            das_patched_pred = pair_result["patched_predictions"]
            das_patched_track = pair_result["patched_track"]
            destination_pred = pair_result["destination_predictions"]
            destination_track = pair_result["destination_track"]

            results.append(
                ValidationResult_on_sample_pair(
                    source=source,
                    destination=destination,
                    patched_predictions=das_patched_pred,
                    patched_track=das_patched_track,
                    clean_predictions=destination_pred,
                    clean_track=destination_track,
                    track_tokens=track_tokens,
                )
            )

        # summarize the validation results
        track_token_types = ["clean_obj", "target_obj"]
        summary = {}
        for token_type in track_token_types:
            print(f"{token_type}")
            ranks = {"clean": [], "patch": []}
            logits = {"clean": [], "patch": []}
            for result in results:
                target_tok_id = result.track_tokens[token_type]
                clean_rank, clean_pred = result.clean_track[target_tok_id]
                patch_rank, patch_pred = result.patched_track[target_tok_id]
                ranks["clean"].append(clean_rank)
                ranks["patch"].append(patch_rank)
                logits["clean"].append(clean_pred.logit)
                logits["patch"].append(patch_pred.logit)

            attr = {
                "rank": ranks,
                "logit": logits,
            }
            for key in attr:
                clean = np.array(attr[key]["clean"])
                patch = np.array(attr[key]["patch"])
                delta = patch - clean
                print(
                    f"{key}: clean {clean.mean():.2f} ± {clean.std():.2f} -> patch {patch.mean():.2f} ± {patch.std():.2f} | delta {delta.mean():.2f} ± {delta.std():.2f}"
                )

            summary[token_type] = attr

        top_1_accuracy = 0
        for result in results:
            target_tok_id = result.destination.metadata["track_type_obj_token_id"]
            patched_track = result.patched_track
            if (
                patched_track[list(patched_track.keys())[0]][1].token_id
                == target_tok_id
            ):
                top_1_accuracy += 1

        top_1_accuracy = top_1_accuracy / len(results)
        summary["top_1_accuracy"] = top_1_accuracy
        print(f"Top-1 accuracy: {top_1_accuracy:.3f}")

        validation_results = ValidationResults(
            layer=layer,
            projection_dim="full",
            results=results,
            summary=summary,
        )
        with open(os.path.join(layer_save_dir, "validation_results.json"), "w") as f:
            f.write(validation_results.to_json(indent=4))
        logger.info("#" * 100)


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


if __name__ == "__main__":

    ##################################################################
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    ##################################################################

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
        "--save_dir",
        type=str,
        default="selection/das_projections/sweep",
        help="Directory to save projections and validation results",
    )

    parser.add_argument(
        "--train_path",
        type=str,
        default=os.path.join(
            env_utils.DEFAULT_RESULTS_DIR,
            # "selection/samples/train/gemma-2-27b-it/select_one/objects",
            "selection/samples/train/Qwen2.5-32B-Instruct/select_one/objects",
        ),
    )

    parser.add_argument(
        "--validation_path",
        type=str,
        default=os.path.join(
            env_utils.DEFAULT_RESULTS_DIR,
            # "selection/samples/validation/gemma-2-27b-it/select_one/objects",
            "selection/samples/validation/Qwen2.5-32B-Instruct/select_one/objects",
        ),
    )

    parser.add_argument(
        "--projection_path",
        type=str,
        default=None,
        help="Will load projections from this path, if specified",
    )

    parser.add_argument(
        "--train_limit",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--validation_limit",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--token_indices",
        type=int,
        nargs="+",
        default=[-3, -2, -1],
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[25],
    )

    parser.add_argument(
        "--proj_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--full_rank",
        action="store_true",
        help="Whether to use full rank projections | patching the whole vector",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(f"Arguments: {args}")
    token_mapping = {i: i for i in args.token_indices}

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
        # attn_implementation="eager",
    )
    layers = (
        mt.layer_names[::2]
        if args.layers[0] == -1
        else [mt.layer_name_format.format(layer_idx) for layer_idx in args.layers]
    )
    # layers = [mt.layer_name_format.format(layer_idx) for layer_idx in range(40, 80, 2)]

    # loading the datasets
    train_set = load_dataset(path=args.train_path, limit=args.train_limit)
    logger.info(f"Loaded {len(train_set)} training samples")
    validation_set = load_dataset(
        path=args.validation_path, limit=args.validation_limit
    )
    logger.info(f"Loaded {len(validation_set)} validation samples")

    if args.full_rank:
        logger.info("Running residual stream patching sweep")

        # Setup cache directory
        save_dir = os.path.join(
            env_utils.DEFAULT_RESULTS_DIR,
            args.save_dir,
            mt.name.split("/")[-1],
            "full_rank",
        )
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Saving results to {save_dir}")
        residual_patching_sweep(
            mt=mt,
            validation_set=validation_set,
            layers=layers,
            token_mapping=token_mapping,
            save_dir=save_dir,
        )

    else:
        logger.info("Running DAS projection sweep")

        # Setup cache directory
        save_dir = os.path.join(
            env_utils.DEFAULT_RESULTS_DIR,
            args.save_dir,
            mt.name.split("/")[-1],
            str(args.proj_dim),
        )
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Saving results to {save_dir}")
        das_sweep(
            mt=mt,
            train_set=train_set,
            validation_set=validation_set,
            layers=layers,
            token_mapping=token_mapping,
            projection_dim=args.proj_dim,
            epochs=args.epochs,
            learning_rate=1e-3,
            ortho_reg=1e-1,
            batch_size=args.batch_size,
            save_dir=save_dir,
            projection_path=args.projection_path,
        )

    logger.info("#" * 100)
    logger.info("All done!")
