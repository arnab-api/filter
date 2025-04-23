import logging
import os
from typing import Optional

import numpy as np
import pytorch_lightning as lightning
import torch
import wandb
from transformers import AdamW, get_linear_schedule_with_warmup

from src.utils import env_utils
from src.utils.typing import Model, Tokenizer

logger = logging.getLogger(__name__)


# Create a LightningModule for training the model
class LM_FineTuner(lightning.LightningModule):
    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        regularizer_lambda: float = 0.1,
        reg_docs: Optional[list[str]] = None,
        max_length: int = 512,
        batch_size: int = 8,
        save_path: str = "ft_checkpoints",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["reg_docs", "tokenizer", "model"])

        self.model = model
        self.tokenizer = tokenizer

        # Create save directory if it doesn't exist
        save_path = os.path.join(env_utils.DEFAULT_RESULTS_DIR, save_path)
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

        # Prepare regularization documents
        self.reg_docs = reg_docs
        self.reg_encodings = []
        self.original_reg_losses = []
        if reg_docs:
            self.reg_encodings = []
            for doc in reg_docs:
                encoding = self.tokenizer(
                    doc, truncation=True, max_length=max_length, return_tensors="pt"
                )
                self.reg_encodings.append(
                    {
                        "input_ids": encoding.input_ids[0],
                        "attention_mask": encoding.attention_mask[0],
                    }
                )

            # Calculate and store initial losses for all reg docs
            # We'll do this in batches to be efficient
            num_batches = (len(self.reg_encodings) + batch_size - 1) // batch_size

            # ** CAUTION **
            #! current regularization loss just considers the overall loss
            #! we would like to make as little change to the representaion/logit distribution as well
            #! maybe it should also be MSE on some of the representaion and/or KL on the logits
            # TODO: first check the LM behavior with the current setup. No need to overcomplicate it if things just seem to work.

            # Process all regularization documents in batches
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.reg_encodings))

                # Prepare batch
                batch_input_ids = []
                batch_attention_mask = []

                for idx in range(start_idx, end_idx):
                    batch_input_ids.append(self.reg_encodings[idx]["input_ids"])
                    batch_attention_mask.append(
                        self.reg_encodings[idx]["attention_mask"]
                    )

                # Stack to create batches
                batch_input_ids = torch.stack(batch_input_ids)
                batch_attention_mask = torch.stack(batch_attention_mask)

                # Calculate initial loss for this batch
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        labels=batch_input_ids,
                    )

                    # Store individual losses for each document in the batch
                    if hasattr(outputs, "loss_per_example"):
                        # If the model outputs per-example losses, use those
                        batch_losses = outputs.loss_per_example
                    else:
                        # Otherwise use the same average loss for all in this batch
                        batch_losses = [outputs.loss.item()] * (end_idx - start_idx)

                    # Extend our list of original losses
                    self.original_reg_losses.extend(batch_losses)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        # Forward pass for normal training data
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Get main training loss (next word prediction)
        train_loss = outputs.loss

        # Initialize regularization loss
        reg_loss = 0.0

        # Calculate regularization loss on reg_docs if provided
        if self.reg_docs and self.hparams.regularizer_lambda > 0:
            # Sample a batch of regularization documents
            batch_size = batch["input_ids"].size(0)  # Get the current batch size
            num_reg_samples = min(batch_size, len(self.reg_encodings))

            if num_reg_samples > 0:
                # Sample indices without replacement
                indices = np.random.choice(
                    len(self.reg_encodings), num_reg_samples, replace=False
                )

                # Prepare batched tensors
                reg_input_ids = []
                reg_attention_mask = []
                orig_losses = []

                for idx in indices:
                    reg_input_ids.append(self.reg_encodings[idx]["input_ids"])
                    reg_attention_mask.append(self.reg_encodings[idx]["attention_mask"])
                    orig_losses.append(self.original_reg_losses[idx])

                # Stack to create batches
                reg_input_ids = torch.stack(reg_input_ids).to(self.device)
                reg_attention_mask = torch.stack(reg_attention_mask).to(self.device)
                orig_losses = torch.tensor(orig_losses, device=self.device)

                # Calculate current loss on the regularization documents
                current_outputs = self.model(
                    input_ids=reg_input_ids,
                    attention_mask=reg_attention_mask,
                    labels=reg_input_ids,
                )
                current_loss = current_outputs.loss

                # For batch-level loss, we need to compare against the average original loss
                orig_loss_avg = orig_losses.mean()

                # Regularization: penalize if current loss is higher than original
                reg_loss = torch.max(
                    torch.zeros_like(current_loss), current_loss - orig_loss_avg
                )

        # Combine the losses
        total_loss = train_loss + self.hparams.regularizer_lambda * reg_loss

        # Increment global step counter
        self.global_step_count += 1

        # Log metrics using PyTorch Lightning's self.log
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.hparams.regularizer_lambda > 0:
            self.log("reg_loss", reg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Explicitly log to wandb
        wandb.log(
            {
                "train/loss": train_loss.item(),
                "train/reg_loss": (
                    reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
                ),
                "train/total_loss": total_loss.item(),
                "train/learning_rate": self.optimizers().param_groups[0]["lr"],
                "global_step": self.global_step_count,
            }
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Forward pass for validation data
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        val_loss = outputs.loss
        perplexity = torch.exp(val_loss)

        # Log metrics using PyTorch Lightning's self.log
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val_perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=True
        )

        # Explicitly log to wandb at step level
        wandb.log(
            {
                "val/loss": val_loss.item(),
                "val/perplexity": perplexity.item(),
                "global_step": self.global_step_count,
            }
        )

        # Return dictionary for epoch-end processing
        return {"val_loss": val_loss, "val_perplexity": perplexity}

    def configure_optimizers(self):
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_epoch_end(self):
        # Save model after each epoch
        epoch = self.current_epoch
        save_dir = os.path.join(self.save_path, f"epoch_{epoch}")
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
