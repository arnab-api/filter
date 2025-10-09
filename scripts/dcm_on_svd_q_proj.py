import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Literal, Union

import baukit
import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from tqdm.auto import tqdm

from src.models import ModelandTokenizer
from src.selection.data import (
    CounterFactualSamplePair,
    CountingSample,
    SelectionSample,
    YesNoSample,
)
from src.selection.optimization import (
    get_optimal_component_mask,
    validate_low_rank_svd_bases_on_sample_pair,
)
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import SVD, PathLike, PredictedToken

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


def validate(
    mt: ModelandTokenizer,
    validation_set: list[SelectionSample, SelectionSample],
    projections: dict[str, torch.Tensor],
    save_dir: PathLike,
    file_name="validation_results",
    token_indices: list[int] = [-1],
):
    os.makedirs(save_dir, exist_ok=True)

    validation_results = []
    for destination, source in tqdm(validation_set):
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

        pair_result = validate_low_rank_svd_bases_on_sample_pair(
            mt=mt,
            source_sample=source,
            destination_sample=destination,
            projections=projections,
            token_indices=token_indices,
            must_track_tokens=list(track_tokens.values()),
            return_clean_predictions=True,
            debug=False,
        )
        patched_pred = pair_result["patched_predictions"]
        patched_track = pair_result["patched_track"]
        destination_pred = pair_result["destination_predictions"]
        destination_track = pair_result["destination_track"]

        validation_results.append(
            ValidationResult_on_sample_pair(
                source=source,
                destination=destination,
                patched_predictions=patched_pred,
                patched_track=patched_track,
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
    for result in validation_results:
        target_tok_id = result.destination.metadata["track_type_obj_token_id"]
        patched_track = result.patched_track
        if patched_track[list(patched_track.keys())[0]][1].token_id == target_tok_id:
            top_1_accuracy += 1

    top_1_accuracy = top_1_accuracy / len(validation_results)
    summary["top_1_accuracy"] = top_1_accuracy
    print(f"Top-1 accuracy: {top_1_accuracy:.3f}")

    validation_results = ValidationResults(
        layer="svd_q_proj_all_layers",
        projection_dim="svd_mask",
        results=validation_results,
        summary=summary,
    )
    with open(os.path.join(save_dir, f"{file_name}.json"), "w") as f:
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


def calculate_and_save_svd(
    path: PathLike,
    mt: ModelandTokenizer,
):
    n_heads = mt.config.num_attention_heads
    head_dim = baukit.get_module(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim
    os.makedirs(path, exist_ok=True)
    for layer_idx in tqdm(range(mt.n_layer)):
        attn_block_name = mt.attn_module_name_format.format(layer_idx)
        attn_block = baukit.get_module(mt._model, attn_block_name)
        q_proj_per_head = attn_block.q_proj.weight.T.view(
            attn_block.q_proj.in_features, n_heads, head_dim
        )
        for head_idx in range(n_heads):
            q_proj_head = q_proj_per_head[:, head_idx, :]
            svd = SVD.calculate(q_proj_head)
            with open(os.path.join(path, f"{layer_idx}_{head_idx}.pt"), "wb") as f:
                torch.save(svd, f)


def load_basis_directions(
    path: PathLike, from_layer: int, to_layer: int
) -> dict[tuple[int, int], torch.Tensor]:
    q_proj_basis_directions = {}
    head_dim = baukit.get_module(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim
    n_embd = mt.n_embd
    n_heads = mt.config.num_attention_heads

    for layer_idx in tqdm(range(mt.n_layer)):
        if layer_idx < from_layer or layer_idx >= to_layer:
            continue
        for head_idx in range(n_heads):
            with open(os.path.join(path, f"{layer_idx}_{head_idx}.pt"), "rb") as f:
                svd = torch.load(f, weights_only=False)
            assert isinstance(svd, SVD)
            assert svd.U.shape == (n_embd, head_dim)
            assert svd.S.shape == (head_dim,)
            assert svd.V.shape == (head_dim, head_dim)

            q_proj_basis_directions[(layer_idx, head_idx)] = svd.V.T.to(
                mt.dtype
            )  #! the transpose here is important!

    return q_proj_basis_directions


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
        "--svd_path",
        type=str,
        default=None,
        help="Will load svd from this path, if specified",
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
        "--from_layer",
        type=int,
        default=None,
        help="Specify layer range (from)",
    )

    parser.add_argument(
        "--to_layer",
        type=int,
        default=None,
        help="Specify layer range (to)",
    )

    parser.add_argument(
        "--token_indices",
        type=int,
        nargs="+",
        default=[-1],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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

    from_layer = args.from_layer or 0
    to_layer = args.to_layer or mt.n_layer

    # calculating/loading the svds
    if args.svd_path is not None:
        logger.info(f"Loading svd from {args.svd_path}")
        q_proj_basis_directions = load_basis_directions(
            args.svd_path, from_layer, to_layer
        )
    else:
        logger.info("Calculating and saving svd")
        head_svd_save_path = os.path.join(
            env_utils.DEFAULT_RESULTS_DIR,
            args.save_dir,
            mt.name.split("/")[-1],
            "svd",
        )
        if not os.path.exists(head_svd_save_path):
            os.makedirs(head_svd_save_path, exist_ok=True)
            # calculate only if the path does not exist, try to load otherwise
            calculate_and_save_svd(path=head_svd_save_path, mt=mt)
        q_proj_basis_directions = load_basis_directions(
            head_svd_save_path, from_layer, to_layer
        )

    # loading the datasets
    train_set = load_dataset(path=args.train_path, limit=args.train_limit)
    logger.info(f"Loaded {len(train_set)} training samples")
    validation_set = load_dataset(
        path=args.validation_path, limit=args.validation_limit
    )
    logger.info(f"Loaded {len(validation_set)} validation samples")

    logger.info("Running DCM on SVD Q-projections")

    # Setup cache directory
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        args.save_dir,
        mt.name.split("/")[-1],
    )
    os.makedirs(save_dir, exist_ok=True)

    # training
    projections, masks, losses = get_optimal_component_mask(
        mt=mt,
        train_set=train_set,
        q_proj_basis_directions=q_proj_basis_directions,
        query_indices=args.token_indices,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-2,
        lamb=1e-4,
        save_step=17,
    )

    # validating
    results = validate(
        mt=mt,
        validation_set=validation_set,
        projections=projections,
        save_dir=save_dir,
        file_name=f"validation_results_{from_layer}_{to_layer}_layers",
        token_indices=args.token_indices,
    )

    logger.info("#" * 100)
    logger.info("All done!")
