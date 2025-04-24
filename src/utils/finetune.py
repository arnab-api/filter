import copy
import logging
import os
from typing import Optional

import numpy as np
import pytorch_lightning as lightning
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.functional import free_gpu_cache
from src.utils import env_utils
from src.utils.typing import Model, Tokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, docs, tokenizer, max_length=512):
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        text = self.docs[idx]

        # Tokenize the text with special tokens
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
        )

        # Get input_ids and create labels (shifted to the right for next token prediction)
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# Create a LightningModule for training the model
class LM_FineTuner(lightning.LightningModule):
    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
        regularization_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        regularizer_lambda: float = 0.1,
        save_path: str = "ft_checkpoints",
        save_interval: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["regularization_dataloader", "tokenizer", "model"]
        )

        self.model = model
        self.tokenizer = tokenizer

        # Create save directory if it doesn't exist
        save_path = os.path.join(env_utils.DEFAULT_RESULTS_DIR, save_path)
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

        # Set saving interval
        self.save_interval = save_interval
        self.global_step_count = 0

        # Prepare regularization documents
        if regularization_dataloader is not None:
            self.cached_reg_info = []
            regularization_dataloader = copy.deepcopy(
                regularization_dataloader
            )  # because we will need that later while training

            for cur_batch in tqdm(regularization_dataloader):
                # Calculate initial loss for this batch
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=cur_batch["input_ids"].to(model.device),
                        attention_mask=cur_batch["attention_mask"].to(model.device),
                        labels=cur_batch["input_ids"].to(model.device),
                    )

                    # ** CAUTION **
                    #! current regularization loss just considers the overall loss
                    #! we would like to make as little change to the representaion/logit distribution as well
                    #! maybe it should also be MSE on some of the representaion and/or KL on the logits
                    # TODO: first check the LM behavior with the current setup. No need to overcomplicate it if things just seem to work.
                    batch_size = cur_batch["input_ids"].size(0)
                    loss = outputs.loss / batch_size

                self.cached_reg_info.append(
                    {
                        "input_ids": cur_batch["input_ids"],
                        "attention_mask": cur_batch["attention_mask"],
                        "loss": loss,
                    }
                )

            free_gpu_cache()

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
        batch_size = batch["input_ids"].size(0)  # Get the current batch size
        train_loss = outputs.loss / batch_size

        # Initialize regularization loss
        reg_loss = 0.0

        if hasattr(self, "cached_reg_info"):
            # randomly select one of the cached regularization documents
            reg_doc = np.random.choice(self.cached_reg_info)
            batch_input_ids = reg_doc["input_ids"].to(self.model.device)
            batch_attention_mask = reg_doc["attention_mask"].to(self.model.device)
            orig_loss = reg_doc["loss"].to(self.model.device)

            # Calculate the current loss on the regularization document
            current_outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_input_ids,
            )
            current_loss = current_outputs.loss / batch_size

            # For batch-level loss, we need to compare against the average original loss
            reg_loss = torch.max(
                torch.zeros_like(current_loss), current_loss - orig_loss
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

        # # log to wandb
        # wandb.log(
        #     {
        #         "train/loss": train_loss.item(),
        #         "train/reg_loss": (
        #             reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
        #         ),
        #         "train/total_loss": total_loss.item(),
        #         "train/learning_rate": self.optimizers().param_groups[0]["lr"],
        #         "global_step": self.global_step_count,
        #     }
        # )

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

        # # Explicitly log to wandb at step level
        # wandb.log(
        #     {
        #         "val/loss": val_loss.item(),
        #         "val/perplexity": perplexity.item(),
        #         "global_step": self.global_step_count,
        #     }
        # )

        # Return dictionary for epoch-end processing
        return {"val_loss": val_loss, "val_perplexity": perplexity}

    def configure_optimizers(self):
        tunable_param_dict = {
            name: param for name, param in self.model.named_parameters()
        }

        # only tune the laers
        # don't tune the embeddings, don't tune the lm head and the final layer norm.
        remove_modules = ["model.embed_tokens.weight"]
        for module_name in tunable_param_dict.keys():
            if module_name.startswith("model.layers.") == False:
                remove_modules.append(module_name)

        for rm in remove_modules:
            if rm in tunable_param_dict:
                tunable_param_dict.pop(rm)

        # Create optimizer
        optimizer = AdamW(
            list(tunable_param_dict.values()),
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

    def on_train_epoch_end(self):
        free_gpu_cache()
        # Save model at specified intervals. Only keep the latest one to save space.
        logger.info(f"Epoch {self.current_epoch} | {self.save_interval=}")
        epoch = self.current_epoch
        if epoch % self.save_interval == 0:
            save_dir = os.path.join(self.save_path, f"epoch_{epoch}")

            # Find and remove the previous checkpoint if it exists
            if epoch >= self.save_interval:  # Only if this isn't the first checkpoint
                prev_epoch = epoch - self.save_interval
                prev_save_dir = os.path.join(self.save_path, f"epoch_{prev_epoch}")
                if os.path.exists(prev_save_dir):
                    logger.info(f"Removing previous checkpoint at {prev_save_dir}")
                    import shutil

                    shutil.rmtree(prev_save_dir)

            logger.info(f"Saving model checkpoint at epoch {epoch} to {save_dir}")
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
