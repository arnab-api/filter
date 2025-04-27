import argparse
import datetime
import json
import logging
import os
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import wandb
from src.functional import free_gpu_cache
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.training_utils import TextDataset, TrainableLM, Trainer

logger = logging.getLogger(__name__)


def run_finetuning(
    mt: ModelandTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    reg_loader: Optional[DataLoader] = None,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    regularizer_lambda: float = 0.1,
    warmup_steps: int = 0,
    max_epochs: int = 5,
    save_path: str = "ft_checkpoints_acc",
    save_interval: int = 30,
    keep_checkpoints: List[int] = None,
    memory_cleaner_threshold: float = 0.7,
):
    """
    Fine-tune a language model with optional regularization using Hugging Face Accelerate.

    Args:
        mt: ModelandTokenizer object containing the model and tokenizer
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        reg_loader: Optional DataLoader for regularization data
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        regularizer_lambda: Coefficient for regularization loss term
        warmup_steps: Number of warmup steps for learning rate scheduler
        max_epochs: Maximum number of training epochs
        save_path: Path to save model checkpoints
        save_interval: Interval to save model checkpoints
        keep_checkpoints: List of specific epochs to keep checkpoints for
        memory_cleaner_threshold: Threshold for GPU memory cleaning (0-1)
    """
    if keep_checkpoints is None:
        keep_checkpoints = []

    # Initialize wandb run name
    run_name = mt.name.split("/")[-1]

    trainable_lm = TrainableLM(
        model=mt._model,
        tokenizer=mt.tokenizer,
        regularization_dataloader=reg_loader,
        regularizer_lambda=regularizer_lambda,
    )

    trainer = Trainer(
        trainable=trainable_lm,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        num_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        save_path=os.path.join(save_path, run_name),
        save_interval=save_interval,
        keep_checkpoints=keep_checkpoints,
        remove_old_checkpoints=True,
        memory_cleaner_threshold=memory_cleaner_threshold,
        log_to_wandb=True,
    )

    # Start training
    logger.info("Starting fine-tuning process")
    trainer.train()

    logger.info(f"Fine-tuning complete")

    return mt


def prepare_datasets(
    train_docs_path: str,
    tokenizer,
    reg_docs_dataset: Optional[str] = "NeelNanda/wiki-10k",
    reg_limit: int = 1000,
    batch_size: int = 4,
    regularizer_lambda: float = 0.1,
    train_split_ratio: float = 0.8,
    repeat: int = 1,
):
    """
    Prepare datasets and dataloaders for fine-tuning.

    Args:
        train_docs_path: Path to JSON file containing training documents
        tokenizer: Tokenizer to use for preparing datasets
        reg_docs_dataset: HuggingFace dataset identifier for regularization documents
        reg_limit: Number of regularization documents to use
        batch_size: Batch size for training
        regularizer_lambda: Coefficient for regularization loss term (used to determine if reg_loader is needed)
        train_split_ratio: Ratio for train/validation split of the training data
        repeat: Number of times to repeat the training documents

    Returns:
        Tuple of (train_loader, val_loader, reg_loader)
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load regularization documents if needed
    reg_loader = None
    if reg_docs_dataset and regularizer_lambda > 0:
        logger.info(f"Loading regularization dataset: {reg_docs_dataset}")
        regularization_docs = load_dataset(reg_docs_dataset)
        indices = np.random.choice(
            len(regularization_docs["train"]), size=reg_limit, replace=False
        ).tolist()

        regularization_docs = [regularization_docs["train"][i]["text"] for i in indices]
        logger.info(f"{len(regularization_docs)=}")
        regularization_ds = TextDataset(docs=regularization_docs, tokenizer=tokenizer)
        reg_loader = DataLoader(
            regularization_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    # Load training documents
    logger.info(f"Loading training documents from {train_docs_path}")
    finetune_docs = []
    with open(os.path.join(env_utils.DEFAULT_DATA_DIR, train_docs_path), "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict) and "docs" in data[0]:
            # Handle structure like synthetic_entities.json
            for item in data:
                finetune_docs.extend(item["docs"])
        else:
            # Handle flat list of documents
            finetune_docs = data
    else:
        # Handle single object with docs field
        if "docs" in data:
            finetune_docs = data["docs"]
        else:
            raise ValueError(f"Unsupported document format in {train_docs_path}")

    finetune_docs = finetune_docs * repeat
    logger.info(f"{len(finetune_docs)=}")

    # Train/val split
    train_split = int(train_split_ratio * len(finetune_docs))
    np.random.shuffle(finetune_docs)

    train_ds = TextDataset(docs=finetune_docs[:train_split], tokenizer=tokenizer)
    val_ds = TextDataset(docs=finetune_docs[train_split:], tokenizer=tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
    )

    return train_loader, val_loader, reg_loader


if __name__ == "__main__":
    ##################################################################################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    ##################################################################################################
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with Accelerate"
    )
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
            "Qwen/Qwen2.5-14B",
        ],
        default="meta-llama/Llama-3.2-3B",
        help="Model identifier from HuggingFace or local path",
    )

    parser.add_argument(
        "--train_docs",
        type=str,
        default="synthetic_entities.json",
        help="Path to training documents JSON file, relative to data directory",
    )

    parser.add_argument(
        "--reg_dataset",
        type=str,
        default="NeelNanda/wiki-10k",
        help="HuggingFace dataset identifier for regularization documents",
    )

    parser.add_argument(
        "--reg_limit",
        type=int,
        default=1000,
        help="Number of regularization documents to use",
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="Weight decay for optimizer"
    )

    parser.add_argument(
        "--regularizer_lambda",
        type=float,
        default=0.1,
        help="Coefficient for regularization loss term",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler",
    )

    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of training epochs"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="finetuned_models",
        help="Path to save model checkpoints",
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=30,
        help="Interval to save model checkpoints (in epochs)",
    )

    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        nargs="+",
        default=[50, 70],
        help="List of specific epochs to keep checkpoints for",
    )

    parser.add_argument(
        "--memory_cleaner_threshold",
        type=float,
        default=0.7,
        help="Threshold for GPU memory cleaning (0-1)",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run for Weights & Biases (defaults to model name if None)",
    )

    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Datatype for model",
    )

    parser.add_argument(
        "--train_split_ratio",
        type=float,
        default=0.8,
        help="Ratio for train/validation split of the training data",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of times to repeat the training documents",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(f"Arguments: {args}")

    # Map string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    torch_dtype_obj = dtype_map.get(args.torch_dtype, torch.bfloat16)

    # Initialize model and tokenizer
    logger.info(f"Loading model: {args.model}")
    mt = ModelandTokenizer(
        model_key=args.model,
        torch_dtype=torch_dtype_obj,
    )

    # Prepare datasets and dataloaders
    train_loader, val_loader, reg_loader = prepare_datasets(
        train_docs_path=args.train_docs,
        tokenizer=mt.tokenizer,
        reg_docs_dataset=args.reg_dataset,
        reg_limit=args.reg_limit,
        batch_size=args.batch_size,
        regularizer_lambda=args.regularizer_lambda,
        train_split_ratio=args.train_split_ratio,
        repeat=args.repeat,
    )

    # Initialize wandb for logging
    run_name = args.run_name if args.run_name else args.model.split("/")[-1]

    wandb.init(
        entity="reasoning-iterp",
        project="connections",
        name=run_name,
    )

    # Run finetuning
    model = run_finetuning(
        mt=mt,
        train_loader=train_loader,
        val_loader=val_loader,
        reg_loader=reg_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        regularizer_lambda=args.regularizer_lambda,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        save_path=args.save_path,
        save_interval=args.save_interval,
        keep_checkpoints=args.keep_checkpoints,
        memory_cleaner_threshold=args.memory_cleaner_threshold,
    )

    # Close wandb run
    wandb.finish()
