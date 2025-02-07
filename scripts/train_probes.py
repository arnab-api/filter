import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import transformers

from src.functional import detensorize
from src.probing.utils import load_probing_activations
from src.probing.analysis import LinearProbe
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)


logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


def train_and_save_probes(
    model_name: str,
    latent_root: str = "probing_latents",
    token_idx: int = -1,
    layer_name_format: str = "model.layers.{}",
    layer_indices: list[int] = [0, 11, 23],
    limit: int = None,
    validation_split: float = 0.3,
    batch_size: int = 1024,
    epochs: int = 500,
    save_dir: str = "linear_probes",
):
    model_name = model_name.split("/")[-1]
    latent_root = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, "probing_latents", model_name
    )
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, save_dir, model_name, str(token_idx)
    )
    os.makedirs(save_dir, exist_ok=True)

    layers = [layer_name_format.format(i) for i in layer_indices]
    cls_activations = load_probing_activations(
        latent_root=latent_root,
        token_query_pos=token_idx,
        layers=layers,
        limit=limit,
    )

    for layer in layers:
        # load and prepare data
        data = []
        for cls, activations in cls_activations.items():
            for act in activations:
                latent = act[layer]
                data.append((latent, cls))

        random.shuffle(data)

        cut_idx = int(len(data) * (1 - validation_split))
        train_data = data[:cut_idx]
        validation_data = data[cut_idx:]

        logger.info(f"{len(train_data)=}, {len(validation_data)=}")

        train_X, train_y = zip(*train_data)
        validation_X, validation_y = zip(*validation_data)

        # train and validate probe
        probe = LinearProbe.from_data(
            acts=torch.stack(train_X),
            labels=train_y,
            validation_set=(torch.stack(validation_X), validation_y),
            batch_size=batch_size,
            epochs=epochs,
        )

        final_val_acc, conf_matrix = probe.validate(
            x=torch.stack(validation_X),
            y=validation_y,
            batch_size=batch_size,
            return_confusion_matrix=True,
        )
        logger.info("-" * 80)
        logger.info(f"{layer=} | {final_val_acc=}")

        # save probe and confusion matrix
        layer_dir = os.path.join(save_dir, layer)
        os.makedirs(layer_dir, exist_ok=True)

        probe.save(os.path.join(layer_dir, "probe.pt"))
        info = dict(
            model=model_name,
            layer=layer,
            token_idx=token_idx,
            val_acc=final_val_acc,
            conf_matrix=conf_matrix,
        )
        info = detensorize(info)

        with open(os.path.join(layer_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

        logger.info("=" * 30 + "\n")


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
        "--token_idx",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--layer_name_format",
        type=str,
        default="model.layers.{}",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 11, 23],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="linear_probes",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(args)

    train_and_save_probes(
        model_name=args.model,
        token_idx=args.token_idx,
        layer_name_format=args.layer_name_format,
        layer_indices=args.layers,
        limit=args.limit,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_dir=args.save_dir,
    )
