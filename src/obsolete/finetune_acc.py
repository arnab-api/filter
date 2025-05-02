import logging
import os
import shutil
from typing import List, Optional

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


class LMTrainer:
    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        regularization_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        regularizer_lambda: float = 0.1,
        save_path: str = "ft_checkpoints",
        save_interval: int = 1,
        keep_checkpoints: List[int] = None,
        remove_old_checkpoints: bool = True,
        memory_cleaner_threshold: float = 0.7,
        num_epochs: int = 1,
        log_to_wandb: bool = True,
    ):
        """
        Initialize a trainer for language models using Hugging Face Accelerate.

        Args:
            model: The model to fine-tune
            tokenizer: The tokenizer for the model
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            regularization_dataloader: Optional dataloader for regularization
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            regularizer_lambda: Coefficient for regularization loss term
            save_path: Path to save model checkpoints
            save_interval: Interval (in epochs) to save model checkpoints
            keep_checkpoints: List of specific epochs to keep checkpoints for
            remove_old_checkpoints: Whether to remove old checkpoints
            memory_cleaner_threshold: Threshold for GPU memory cleaning
            log_to_wandb: Whether to log metrics to Weights & Biases
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.regularization_dataloader = regularization_dataloader

        # Save hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.regularizer_lambda = regularizer_lambda
        self.num_epochs = num_epochs

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

        # Cache regularization documents
        if self.regularization_dataloader is not None:
            self._cache_regularization_docs()

        # Create optimizer and scheduler
        self._setup_optimizer_and_scheduler()

        # Prepare model and dataloaders with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

    def _cache_regularization_docs(self):
        """Cache regularization documents for later use during training."""
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

    def _setup_optimizer_and_scheduler(self):
        """Set up optimizer and learning rate scheduler."""
        # Get tunable parameters
        tunable_param_dict = self._get_tunable_params()

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

        total_steps = max(total_steps, 10000)

        logger.info(f"Settting total training steps: {total_steps}")

        # Create learning rate scheduler
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

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

        if self.log_to_wandb and self.accelerator.is_local_main_process:
            wandb.log(
                {
                    "trainable_params": trainable_params,
                    "non_trainable_params": non_trainable_params,
                    "total_params": total_params,
                }
            )

        return tunable_param_dict

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
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save model and tokenizer
        unwrapped_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

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
            self.model.train()

            # Initialize metrics for this epoch
            train_loss = 0.0
            reg_loss_sum = 0.0
            total_loss_sum = 0.0
            num_batches = 0

            # Progress bar for this epoch
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            # Batch loop
            for batch_idx, batch in enumerate(progress_bar):
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                # Calculate loss
                batch_size = find_batch_size(batch["input_ids"])
                train_batch_loss = outputs.loss / batch_size

                # Handle regularization if needed
                reg_loss = 0.0
                if hasattr(self, "cached_reg_info") and self.regularizer_lambda > 0:
                    # Randomly select a cached regularization document
                    reg_doc = np.random.choice(self.cached_reg_info)

                    # Move to device
                    batch_input_ids = reg_doc["input_ids"].to(self.model.device)
                    batch_attention_mask = reg_doc["attention_mask"].to(
                        self.model.device
                    )
                    orig_loss = reg_doc["loss"].to(self.model.device)

                    # Calculate current loss on regularization document
                    current_outputs = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        labels=batch_input_ids,
                    )
                    current_loss = current_outputs.loss / batch_input_ids.size(0)

                    # Calculate regularization loss (max of 0 and increase in loss)
                    # logger.info(
                    #     f"Regularization {current_loss=} | {orig_loss=} | {current_loss - orig_loss=}"
                    # )
                    reg_loss = torch.max(
                        torch.zeros_like(current_loss), current_loss - orig_loss
                    )

                # Combine losses
                total_loss = train_batch_loss + self.regularizer_lambda * reg_loss

                # Backward pass
                self.accelerator.backward(total_loss)

                # Update parameters
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Update metrics
                train_loss += train_batch_loss.detach()
                reg_loss_sum += (
                    reg_loss.detach()
                    if isinstance(reg_loss, torch.Tensor)
                    else reg_loss
                )
                total_loss_sum += total_loss.detach()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "train_loss": train_batch_loss.item(),
                        "reg_loss": (
                            reg_loss.item()
                            if isinstance(reg_loss, torch.Tensor)
                            else reg_loss
                        ),
                        "total_loss": total_loss.item(),
                    }
                )

                # Log metrics directly to wandb instead of using accelerator.log
                if self.log_to_wandb and self.accelerator.is_local_main_process:
                    wandb.log(
                        {
                            "train/loss": train_batch_loss.item(),
                            "train/reg_loss": (
                                reg_loss.item()
                                if isinstance(reg_loss, torch.Tensor)
                                else reg_loss
                            ),
                            "train/total_loss": total_loss.item(),
                            "train/lr": self.lr_scheduler.get_last_lr()[0],
                            "step": self.global_step,
                        }
                    )

                # Increment global step
                self.global_step += 1

                # Maybe clean up memory
                if batch_idx % 10 == 0:
                    self._maybe_cleanup_memory()

            # End of epoch
            train_loss /= num_batches
            reg_loss_sum /= num_batches
            total_loss_sum /= num_batches

            # Log epoch metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Reg Loss: {reg_loss_sum:.4f} | "
                f"Total Loss: {total_loss_sum:.4f}"
            )

            # Run evaluation
            eval_results = self.evaluate()

            # Log epoch-level metrics directly to wandb
            if self.log_to_wandb and self.accelerator.is_local_main_process:
                logger.info("Logging epoch-level metrics to wandb")
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "epoch/train_loss": train_loss.item(),
                        "epoch/reg_loss": (
                            reg_loss_sum.item()
                            if isinstance(reg_loss_sum, torch.Tensor)
                            else reg_loss_sum
                        ),
                        "epoch/total_loss": total_loss_sum.item(),
                        "epoch/val_loss": eval_results["loss"],
                        "epoch/val_perplexity": eval_results["perplexity"],
                        "step": self.global_step,
                    }
                )

            # Save checkpoint
            self._save_checkpoint(epoch + 1)

            # Clean up memory at end of epoch
            free_gpu_cache()

        # End of training
        # Save final model
        self._save_checkpoint(self.num_epochs, is_final=True)

        logger.info("Training complete!")
        return self.model

    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dict containing evaluation metrics
        """
        self.model.eval()

        eval_loss = 0.0
        num_eval_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process,
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs.loss
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
