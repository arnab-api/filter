import os

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)


# 3. Main training function
def train_language_model(
    train_docs,
    validation_docs,
    reg_docs=None,
    model_name="gpt2",
    batch_size=4,
    max_length=512,
    learning_rate=5e-5,
    weight_decay=0.0,
    regularizer_lambda=0.1,
    num_epochs=3,
    warmup_steps=0,
    save_path="./model_checkpoints",
    project_name="lm-finetuning",
):
    # Initialize wandb
    wandb.init(project=project_name)
    wandb_logger = WandbLogger()

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    train_dataset = TextDataset(train_docs, tokenizer, max_length)
    val_dataset = TextDataset(validation_docs, tokenizer, max_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # Initialize model
    model = GPT2FineTuner(
        model_name_or_path=model_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        regularizer_lambda=regularizer_lambda,
        reg_docs=reg_docs,
        tokenizer=tokenizer,
        max_length=max_length,
        save_path=save_path,
    )

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Close wandb run
    wandb.finish()

    return model


# Usage example
if __name__ == "__main__":
    # Define your documents
    train_docs = ["Your training document 1", "Your training document 2", "..."]
    validation_docs = [
        "Your validation document 1",
        "Your validation document 2",
        "...",
    ]
    reg_docs = [
        "Your regularization document 1",
        "Your regularization document 2",
        "...",
    ]

    # Train the model
    model = train_language_model(
        train_docs=train_docs,
        validation_docs=validation_docs,
        reg_docs=reg_docs,
        model_name="gpt2",
        batch_size=4,
        max_length=512,
        learning_rate=5e-5,
        weight_decay=0.0,
        regularizer_lambda=0.1,
        num_epochs=3,
        warmup_steps=100,
        save_path="./model_checkpoints",
        project_name="lm-finetuning",
    )
