import argparse
import logging
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin

from src.attention import AttentionInformation, get_attention_matrices
from src.functional import detensorize
from src.models import ModelandTokenizer
from src.selection.data import (
    SelectionSample,
    get_random_sample,
    load_people_by_category,
)
from src.tokens import find_token_range, prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import ArrayLike, PathLike

logger = logging.getLogger(__name__)


@dataclass
class SelectionSampleAttn(DataClassJsonMixin):
    sample: SelectionSample
    option_ranges: list[tuple[int, int]]
    attention_pattern: ArrayLike
    value_weighted_attention_pattern: ArrayLike | None = None

    def from_npz(file: np.lib.npyio.NpzFile | PathLike):
        if isinstance(file, PathLike):
            file = np.load(file, allow_pickle=True)

        sample = SelectionSample.from_dict(file["sample"].item())
        option_ranges = file["option_ranges"].tolist()
        attention_pattern = AttentionInformation.from_dict(
            file["attention_pattern"].item()
        )
        value_weighted_attention_pattern = (
            AttentionInformation.from_dict(
                file["value_weighted_attention_pattern"].item()
            )
            if "value_weighted_attention_pattern" in file
            else None
        )
        return SelectionSampleAttn(
            sample=sample,
            option_ranges=option_ranges,
            attention_pattern=attention_pattern,
            value_weighted_attention_pattern=value_weighted_attention_pattern,
        )

    def score_per_option(
        self,
        layer_idx: int,
        head_idx: int,
        query_idx: int = -1,
        value_weighted: bool = False,
        token_idx: Literal["last", "first", "all"] = "all",
    ):
        """Get the attention score for a specific layer and head."""
        attn_row = (
            self.attention_pattern.attention_matrices[layer_idx][head_idx][query_idx]
            if not value_weighted
            else self.value_weighted_attention_pattern.attention_matrices[layer_idx][
                head_idx
            ][query_idx]
        )
        attn_row = (
            torch.Tensor(attn_row)
            if not isinstance(attn_row, torch.Tensor)
            else attn_row
        )
        scores = []
        if token_idx == "all":
            option_ranges = self.option_ranges
        elif token_idx == "last":
            option_ranges = [(end - 1, end) for start, end in self.option_ranges]
        elif token_idx == "first":
            option_ranges = [(start, start + 1) for start, end in self.option_ranges]
        else:
            raise ValueError(
                f"Invalid token_idx: {token_idx}. Must be 'last', 'first', or 'all'."
            )
        for obj_range in option_ranges:
            start, end = obj_range
            scores.append(attn_row[start:end].sum().item())

        return scores

    def resolution_score(
        self,
        layer_idx: int,
        head_idx: int,
        query_idx: int = -1,
        value_weighted: bool = False,
        token_idx: Literal["last", "first", "all"] = "all",
    ):
        """See if the head prefers the answer over other options."""
        score_per_option = self.score_per_option(
            layer_idx=layer_idx,
            head_idx=head_idx,
            query_idx=query_idx,
            value_weighted=value_weighted,
            token_idx=token_idx,
        )
        answer_score = score_per_option[self.sample.obj_idx]
        other_scores = [
            score
            for i, score in enumerate(score_per_option)
            if i != self.sample.obj_idx
        ]
        diff = answer_score - max(other_scores)
        return diff, answer_score, other_scores

    def first_token_score(
        self,
        layer_idx: int,
        head_idx: int,
        query_idx: int = -1,
        value_weighted: bool = False,
    ):
        """Check if the head is looking only at the first token of the object."""
        attn_row = (
            self.attention_pattern.attention_matrices[layer_idx][head_idx][query_idx]
            if not value_weighted
            else self.value_weighted_attention_pattern.attention_matrices[layer_idx][
                head_idx
            ][query_idx]
        )
        attn_row = (
            torch.Tensor(attn_row)
            if not isinstance(attn_row, torch.Tensor)
            else attn_row
        )
        frm, to = self.option_ranges[self.sample.obj_idx]
        first_token_score = attn_row[frm].item()
        other_scores = attn_row[frm + 1 : to].sum().item()
        return first_token_score - other_scores, first_token_score, other_scores


@torch.inference_mode()
def get_attention_pattern_for_selection_sample(
    mt: ModelandTokenizer,
    sample: SelectionSample,
    add_value_weighted: bool = False,
):
    tokenized = prepare_input(
        tokenizer=mt,
        prompts=sample.prompt,
        return_offsets_mapping=True,
        # add_bos_token="qwen" in mt.name.lower(), #! adding a bos token messes with the offsets
    )
    offsets = tokenized.pop("offset_mapping")[0]
    options_ranges = []
    for opt in sample.options:
        start, end = find_token_range(
            string=sample.prompt,
            substring=opt,
            offset_mapping=offsets,
            tokenizer=mt.tokenizer,
        )
        logger.debug(
            f'{opt=} {(start, end)} "{mt.tokenizer.decode(tokenized["input_ids"][0][start:end])}"'
        )
        options_ranges.append((start, end))

    attention_pattern = get_attention_matrices(
        input=tokenized, mt=mt, value_weighted=False
    )
    if add_value_weighted:
        value_weighted_attention_pattern = get_attention_matrices(
            input=tokenized, mt=mt, value_weighted=True
        )
    else:
        value_weighted_attention_pattern = None

    return SelectionSampleAttn(
        sample=sample,
        option_ranges=options_ranges,
        attention_pattern=attention_pattern,
        value_weighted_attention_pattern=value_weighted_attention_pattern,
    )


@torch.inference_mode()
def cache_attention_patterns_for_selection_samples(
    mt: ModelandTokenizer,
    limit: int = 12,
    save_dir: str | None = None,
    category: str = "occupation",
    n_distractors: int = 5,
):

    people_by_category = load_people_by_category(tokenizer=mt.tokenizer)
    os.makedirs(save_dir, exist_ok=True)

    test_head_map = {
        "meta-llama/Llama-3.3-70B-Instruct": (35, 19),
        "Qwen/Qwen2.5-72B-Instruct": (54, 44),
    }

    layer_idx, head_idx = test_head_map.get(mt.name, (0, 0))

    sample_idx = 0
    for sample_idx in range(limit):
        logger.info(f"Processing sample {sample_idx + 1}/{limit}")
        sample = get_random_sample(
            people_by_category=people_by_category,
            mt=mt,
            category=category,
            n_distractors=n_distractors,
            filter_by_lm_prediction=True,
        )
        logger.debug(f"{str(sample)}")
        logger.debug(f"{[str(pred) for pred in sample.prediction]}")

        attn_pattern = get_attention_pattern_for_selection_sample(
            mt=mt,
            sample=sample,
            add_value_weighted=True,
        )

        logger.debug(
            f"{attn_pattern.resolution_score(layer_idx=layer_idx, head_idx=head_idx)=}"
        )
        attn_pattern = detensorize(attn_pattern, to_numpy=True)

        file_path = os.path.join(save_dir, f"sample_{sample_idx:04d}.npz")
        np.savez_compressed(file=file_path, **attn_pattern.__dict__, allow_pickle=True)


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
        default="selection/attention_patterns",
        help="Directory to save test results",
    )

    parser.add_argument(
        "--attr",
        type=str,
        default="occupation",
        help="Attribute Type",
    )

    parser.add_argument(
        "--n_distractors",
        type=int,
        default=5,
        help="Number of distractors to use",
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

    # # fusing the trained deltas
    # SYNTH_DATASET = "64"
    # checkpoint_path = os.path.join(
    #     env_utils.DEFAULT_RESULTS_DIR,
    #     "trained_params",
    #     f"{SYNTH_DATASET}",
    #     "_full__clamp=0.001",
    #     args.model.split("/")[-1],
    # )
    # version = "epoch_1"
    # checkpoint_path = os.path.join(
    #     env_utils.DEFAULT_RESULTS_DIR, checkpoint_path, version
    # )
    # checkpoint_path = os.path.join(checkpoint_path, "trainable_params.pt")
    # loaded_deltas = torch.load(checkpoint_path, map_location="cpu")
    # free_gpu_cache()
    # TrainableLM_delta.fuse_with_model(mt._model, loaded_deltas)

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
        limit=args.limit,
        save_dir=save_dir,
        category=args.attr,
        n_distractors=args.n_distractors,
    )
