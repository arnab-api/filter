import argparse
import json
import logging
import os

import torch
import transformers

from src.models import ModelandTokenizer
from src.selection.data import SelectionSample, get_random_sample_mixed
from src.utils import env_utils, experiment_utils, logging_utils
from src.selection.data import load_people_by_category, load_people_by_category_fakeverse
from src.selection.data import SelectionSample, get_random_sample
from src.functional import predict_next_token, free_gpu_cache
from dataclasses import dataclass
from src.utils.typing import PredictedToken
from src.utils.training_utils import TrainableLM_delta
from dataclasses_json import DataClassJsonMixin


logger = logging.getLogger(__name__)


logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


@dataclass
class SelectionResults(DataClassJsonMixin):
    sample: SelectionSample
    prediction: list[PredictedToken]
    is_correct: bool
    ans_rank: int


@torch.inference_mode()
def test_selection_with_mixed_entities(
    mt: ModelandTokenizer,
    limit: int = 12,
    attribute_type: str = "profession",
    save_dir: str | None = None,
    n_distractors: int = 5,
    save_step: int = 5,
    synth_pivot: bool = True
):
    """Cache last token states for selection samples."""
    real_people_by_category = load_people_by_category(
        tokenizer=mt.tokenizer,
        category=attribute_type,
    )

    synth_people_by_category = load_people_by_category_fakeverse(
        tokenizer=mt.tokenizer,
        category="occupation",
    )

    os.makedirs(save_dir, exist_ok=True)

    results = []
    n_correct = 0
    while len(results) < limit:
        sample = get_random_sample_mixed(
            real_people_by_category=real_people_by_category,
            synth_people_by_category=synth_people_by_category,
            mt=mt,
            n_distractors=n_distractors,
            filter_by_lm_prediction=False,
            exclude_distractor_categories=["news anchor", "journalist", "entrepreneur", "comedian"],
            synth_pivot=synth_pivot,
        )
        prediction, track_ans = predict_next_token(
            mt=mt, inputs=sample.prompt, k=5, token_of_interest=[sample.obj_token_id]
        )
        top_prediction = prediction[0][0]
        if top_prediction.token_id == sample.obj_token_id:
            n_correct += 1

        results.append(
            SelectionResults(
                sample=sample,
                prediction=prediction[0],
                is_correct=top_prediction.token_id == sample.obj_token_id,
                ans_rank=track_ans[0][sample.obj_token_id][0],
            )
        )
        if len(results) % save_step == 0 or len(results) == limit:
            logger.info(
                f"Cached {len(results)} samples so far, accuracy={n_correct / len(results) : .3f}  ({n_correct}/{len(results)})."
            )
            file_name = f"{attribute_type}_synth_subj_results.json" if synth_pivot else f"{attribute_type}_real_subj_results.json"
            with open(os.path.join(save_dir, file_name), "w") as f:
                json.dump(
                    dict(
                        accuracy=n_correct / len(results),
                        samples=[r.to_dict() for r in results],
                    ),
                    f,
                    indent=4,
                    ensure_ascii=False,
                )


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
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
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
        default="selection/test_2_mixed",
        help="Directory to save test results",
    )

    parser.add_argument(
        "--attr",
        type=str,
        default="profession",
        help="Attribute Type",
    )

    parser.add_argument(
        "--n_distractors",
        type=int,
        default=5,
        help="Number of distractors to use",
    )

    parser.add_argument(
        "--save_step",
        type=int,
        default=5,
        help="Save results every N samples",
    )

    parser.add_argument(
        "--synth_pivot",
        type=lambda x: x.lower() in ('true', '1', 'yes', 'on'),
        default=True,
        help="Flag to make synthetic entity the pivot or the target (pass True or False)"
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
    )

    # fusing the trained deltas
    SYNTH_DATASET = "64"
    checkpoint_path = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        "trained_params",
        f"{SYNTH_DATASET}",
        "_full__clamp=0.001",
        args.model.split("/")[-1],
    )
    version = "epoch_1"
    checkpoint_path = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, checkpoint_path, version
    )
    checkpoint_path = os.path.join(checkpoint_path, "trainable_params.pt")
    loaded_deltas = torch.load(checkpoint_path, map_location="cpu")
    free_gpu_cache()
    TrainableLM_delta.fuse_with_model(mt._model, loaded_deltas)

    # Setup cache directory
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        args.save_dir,
        mt.name.split("/")[-1],
    )
    os.makedirs(save_dir, exist_ok=True)

    test_selection_with_mixed_entities(
        mt=mt,
        limit=args.limit,
        attribute_type=args.attr,
        save_dir=save_dir,
        n_distractors=args.n_distractors,
        save_step=args.save_step,
        synth_pivot=args.synth_pivot,
    )
