import copy
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import find_batch_size
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import wandb
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


def clean_up_grad(module: torch.nn.Module):
    """
    Clean up gradients for all parameters in the module.
    """
    for param in module.parameters():
        if param.grad is not None:
            param.grad = None


class Trainable:
    def __init__(self, model: torch.nn.ModuleDict | Model, **kwargs):
        raise NotImplementedError("should be implemented in child class")

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("should be implemented in child class")

    def get_current_loss(self, *args, **kwargs) -> tuple[float, dict] | float:
        """
        Get the current loss value and additional information.
        """
        raise NotImplementedError("should be implemented in child class")

    def save(self, path: str):
        raise NotImplementedError("should be implemented in child class")

    def _get_tunable_params(self):
        return self.model.parameters()

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()


class TrainableLM(Trainable):
    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
        regularization_dataloader=None,
        regularizer_lambda=0.1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.regularization_dataloader = regularization_dataloader
        self.regularizer_lambda = regularizer_lambda
        self.cached_reg_info = None
        self._cache_regularization_docs()

    @torch.inference_mode()
    def _cache_regularization_docs(self):
        """
        Cache regularization documents for later use during training.
        """
        self.cached_reg_info = []

        logger.info("Caching regularization documents...")
        for cur_batch in tqdm(self.regularization_dataloader):
            # Calculate initial loss for this batch
            with torch.no_grad():
                cur_batch = {k: v.to(self.model.device) for k, v in cur_batch.items()}
                outputs = self.model(
                    input_ids=cur_batch["input_ids"],
                    attention_mask=cur_batch["attention_mask"],
                    labels=cur_batch["input_ids"],
                )

                batch_size = find_batch_size(cur_batch["input_ids"])
                loss = outputs.loss / batch_size

            self.cached_reg_info.append(
                {
                    "input_ids": cur_batch["input_ids"].detach().cpu(),
                    "attention_mask": cur_batch["attention_mask"].detach().cpu(),
                    "loss": loss.detach().cpu(),
                }
            )

        free_gpu_cache()
        logger.info(f"Cached {len(self.cached_reg_info)} regularization batches")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the language model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            labels: Labels for the input (used for calculating loss)

        Returns:
            Loss value
        """
        return self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=(
                attention_mask.to(self.model.device)
                if attention_mask is not None
                else None
            ),
            labels=labels.to(self.model.device) if labels is not None else None,
        )

    def get_current_loss(
        self,
        input_ids,
        attention_mask,
        labels,
        apply_regularization_loss=True,
        **kwargs,
    ) -> tuple[float, dict]:
        """
        Get the current loss value and additional information.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            labels: Labels for the input (used for calculating loss)
            get_reg_loss: Whether to calculate regularization loss

        Returns:
            Tuple containing the loss value and a dictionary with additional information
        """

        for key in kwargs:
            logger.warning(f"Ignoring unexpected keyword argument: {key}={kwargs[key]}")

        # Forward pass
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Calculate loss
        batch_size = find_batch_size(input_ids)
        loss = outputs.loss / batch_size

        loss_dict = {
            "train_loss": loss.detach().item(),
        }

        # Handle regularization if needed
        if (
            apply_regularization_loss
            and hasattr(self, "cached_reg_info")
            and self.regularizer_lambda > 0
        ):
            # Randomly select a cached regularization document
            reg_doc = np.random.choice(self.cached_reg_info)

            # Move to device
            reg_input_ids = reg_doc["input_ids"].to(self.model.device)
            reg_attention_mask = reg_doc["attention_mask"].to(self.model.device)
            orig_loss = reg_doc["loss"].to(self.model.device)

            # Calculate current loss on regularization document
            reg_outputs = self.forward(
                input_ids=reg_input_ids,
                attention_mask=reg_attention_mask,
                labels=reg_input_ids,
            )
            reg_loss = reg_outputs.loss / find_batch_size(reg_input_ids)

            # Calculate regularization loss (max of 0 and increase in loss)
            # logger.info(
            #     f"Regularization {reg_loss=} | {orig_loss=} | {reg_loss - orig_loss=}"
            # )
            reg_loss = torch.max(torch.zeros_like(reg_loss), reg_loss - orig_loss)

            loss_dict["reg_loss"] = reg_loss.detach().item()

            # Combine losses
            loss += self.regularizer_lambda * reg_loss
            loss_dict["total_loss"] = loss.detach().item()

        return loss, loss_dict

    @torch.inference_mode()
    def save(self, path: str, unwrapped_model=None):
        os.makedirs(path, exist_ok=True)
        if unwrapped_model is None:
            self.model.save_pretrained(path)
        else:
            unwrapped_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def _get_tunable_params(self):
        """
        Get the subset of model parameters to optimize.
        For LLaMA models, we only tune the parameters in the layers.
        """
        tunable_param_dict = {
            name: param for name, param in self.model.named_parameters()
        }

        remove_modules = ["model.embed_tokens.weight"]
        for module_name in tunable_param_dict.keys():
            if not module_name.startswith("model.layers."):
                remove_modules.append(module_name)

        for rm in remove_modules:
            if rm in tunable_param_dict:
                tunable_param_dict.pop(rm)

        # Calculate numbers for reporting
        trainable_params = sum(p.numel() for p in tunable_param_dict.values())
        total_params = sum(p.numel() for p in self.model.parameters())
        non_trainable_params = total_params - trainable_params

        # Log parameter counts
        logger.info(f"TRAINABLE PARAMS: {trainable_params / 1e9:.2f}B")
        logger.info(f"NON-TRAINABLE PARAMS: {non_trainable_params / 1e9:.2f}B")
        logger.info(f"TOTAL PARAMS: {total_params / 1e9:.2f}B")

        return tunable_param_dict


class Trainer:
    def __init__(
        self,
        trainable: Trainable,
        # dataloaders
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        # training hyperparameters
        num_epochs: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        # checkpointing
        save_path: str = "ft_checkpoints",
        save_interval: int = 1,
        keep_checkpoints: List[int] = None,
        remove_old_checkpoints: bool = True,
        # memory management
        memory_cleaner_threshold: float = 0.7,
        # wandb logging
        log_to_wandb: bool = True,
    ):
        """
        Initialize a trainer for language models using Hugging Face Accelerate.

        Args:
            model: The model to be trained
            tokenizer: The tokenizer for the model
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            num_epochs: Number of epochs to train for
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            warmup_steps: Number of warmup steps for the learning rate scheduler
            save_path: Path to save checkpoints
            save_interval: Interval (in epochs) to save checkpoints
            keep_checkpoints: List of epochs to keep checkpoints for
            remove_old_checkpoints: Whether to remove old checkpoints when saving new ones
            memory_cleaner_threshold: Threshold for GPU memory utilization cleanup
            log_to_wandb: Whether to log metrics to Weights & Biases
        """
        self.trainable = trainable
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Save hyperparameters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # Setup save path
        self.save_path = os.path.join(env_utils.DEFAULT_RESULTS_DIR, save_path)
        os.makedirs(self.save_path, exist_ok=True)

        # Setup checkpoint saving options
        self.save_interval = save_interval
        self.keep_checkpoints = keep_checkpoints if keep_checkpoints else []
        self.remove_old_checkpoints = remove_old_checkpoints

        # Memory management
        self.memory_cleaner_threshold = memory_cleaner_threshold

        # Logging
        self.log_to_wandb = log_to_wandb
        self.global_step = 0

        # Initialize Accelerator
        self.accelerator = Accelerator()

        # Create optimizer and scheduler
        self._setup_optimizer_and_scheduler()

        # Prepare model and dataloaders with accelerator
        (
            self.trainable.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.trainable.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

    def _setup_optimizer_and_scheduler(self):
        """Set up optimizer and learning rate scheduler."""
        # Get tunable parameters
        tunable_param_dict = self.trainable._get_tunable_params()

        # Create optimizer
        self.optimizer = AdamW(
            list(tunable_param_dict.values()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Calculate total number of training steps
        total_steps = (
            len(self.train_dataloader)
            * self.num_epochs
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

        total_steps = max(total_steps, 100000)

        logger.info(f"Settting total training steps: {total_steps}")

        # Create learning rate scheduler
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

    def _maybe_cleanup_memory(self):
        """Clean up GPU memory if utilization exceeds threshold."""
        if torch.cuda.is_available():
            # Calculate current GPU memory utilization
            allocated = torch.cuda.memory_allocated()
            max_allocated = torch.cuda.max_memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            utilization_pct = allocated / total

            if utilization_pct > self.memory_cleaner_threshold:
                logger.warning(
                    f"GPU Memory Utilization: {utilization_pct:.2f} | "
                    f"Allocated: {allocated / 1e9:.2f} GB | "
                    f"Max Allocated: {max_allocated / 1e9:.2f} GB"
                )
                free_gpu_cache()

    def _save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save a model checkpoint."""
        # Determine if we should save at this epoch
        should_save = (
            is_final
            or (epoch % self.save_interval == 0)
            or (epoch in self.keep_checkpoints)
        )

        if not should_save:
            return

        # Remove previous checkpoint if needed
        if self.remove_old_checkpoints and not is_final:
            prev_epoch = epoch - self.save_interval
            if prev_epoch > 0 and prev_epoch not in self.keep_checkpoints:
                prev_save_dir = os.path.join(self.save_path, f"epoch_{prev_epoch}")
                if os.path.exists(prev_save_dir):
                    logger.info(f"Removing previous checkpoint at {prev_save_dir}")
                    shutil.rmtree(prev_save_dir)

        # Save current checkpoint
        save_dir = os.path.join(
            self.save_path, "final_model" if is_final else f"epoch_{epoch}"
        )
        logger.info(f"Saving model checkpoint to {save_dir}")

        # Unwrap model before saving
        unwrapped_model = self.accelerator.unwrap_model(self.trainable.model)

        # Save model
        self.trainable.save(save_dir, unwrapped_model=unwrapped_model)

    def train(self):
        """
        Train the model for the specified number of epochs.

        Args:
            num_epochs: Number of epochs to train for
        """
        # Log the total number of epochs
        logger.info(f"Starting training for {self.num_epochs} epochs")

        # Training loop
        for epoch in range(self.num_epochs):
            # Set model to training mode
            self.trainable.train_mode()

            # Initialize metrics for this epoch
            total_loss_dict = {}
            num_batches = 0

            # Progress bar for this epoch
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            # Batch loop
            for batch_idx, batch in enumerate(progress_bar):
                loss, loss_info = self.trainable.get_current_loss(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                # Backward pass
                self.accelerator.backward(loss)

                # Update parameters
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Update metrics
                if len(total_loss_dict) == 0:
                    for k in loss_info:
                        total_loss_dict[k] = 0

                for k in loss_info:
                    total_loss_dict[k] += loss_info[k]

                num_batches += 1

                # Log metrics directly to wandb instead of using accelerator.log
                if self.log_to_wandb and self.accelerator.is_local_main_process:
                    wandb_step_report = {
                        "step": self.global_step,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                    for k, v in loss_info.items():
                        wandb_step_report[f"train/{k}"] = v

                    wandb.log(wandb_step_report)

                # Increment global step
                self.global_step += 1

                # Maybe clean up memory
                if batch_idx % 10 == 0:
                    self._maybe_cleanup_memory()

            for k in total_loss_dict:
                total_loss_dict[k] /= num_batches

            # Update progress bar
            progress_bar.set_postfix(total_loss_dict)

            # Log epoch metrics
            loss_log = ""
            for k, v in total_loss_dict.items():
                loss_log += f"{k}: {v:.4f} | "
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} | {loss_log}")

            # Run evaluation
            eval_results = self.evaluate()

            # Log epoch-level metrics directly to wandb
            if self.log_to_wandb and self.accelerator.is_local_main_process:

                wandb_epoch_report = {"epoch": epoch + 1}
                for k, v in total_loss_dict.items():
                    wandb_epoch_report[f"epoch/{k}"] = v

                wandb_epoch_report["epoch/val_loss"] = eval_results["loss"]
                wandb_epoch_report["epoch/val_perplexity"] = eval_results["perplexity"]
                logger.info("Logging epoch-level metrics to wandb", wandb_epoch_report)
                wandb.log(wandb_epoch_report)

            # Save checkpoint
            self._save_checkpoint(epoch + 1)

            # Clean up memory at end of epoch
            free_gpu_cache()

        # End of training
        # Save final model
        self._save_checkpoint(self.num_epochs, is_final=True)

        logger.info("Training complete!")
        return self.trainable

    @torch.inference_mode()
    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dict containing evaluation metrics
        """
        self.trainable.eval_mode()

        eval_loss = 0.0
        num_eval_batches = 0

        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        ):
            loss, _ = self.trainable.get_current_loss(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                apply_regularization_loss=False,
            )

            eval_loss += loss.detach()
            num_eval_batches += 1

        # Average loss
        eval_loss = eval_loss / num_eval_batches

        # Calculate perplexity
        perplexity = torch.exp(eval_loss)

        # Log results
        logger.info(f"Validation Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")

        return {
            "loss": eval_loss.item(),
            "perplexity": perplexity.item(),
        }
