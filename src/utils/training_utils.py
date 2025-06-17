import logging
import math
import os
import shutil
from typing import Any, List, Optional

import baukit
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import find_batch_size
from nnsight import Envoy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import wandb
from src.functional import free_gpu_cache, get_module_nnsight, untuple
from src.models import ModelandTokenizer
from src.utils import env_utils
from src.utils.typing import Model

logger = logging.getLogger(__name__)

from typing import Optional

N_LAYER_DICT = {
    "meta-llama/Llama-3.3-70B-Instruct": 80,
    "Qwen/Qwen2.5-72B-Instruct": 80,
}


def get_device_map(model_name, upto_layer: int, n_gpus: Optional[int] = None):
    n_gpus = torch.cuda.device_count() if n_gpus is None else n_gpus
    if model_name not in N_LAYER_DICT or n_gpus < 2:
        msg = ""
        if model_name not in N_LAYER_DICT:
            msg = f" Model {model_name} not supported"
        if n_gpus < 2:
            msg += f" Only {n_gpus} GPU(s) available"
        msg += "using default device map = `auto`."
        logger.warning(msg)
        return "auto"

    n_layers = N_LAYER_DICT[model_name]
    block_names = [f"model.layers.{i}" for i in range(n_layers)]
    other_modules = ["model.embed_tokens", "model.norm", "model.rotary_emb", "lm_head"]

    device_map = {}

    # 1. Assign other_modules to GPU 0
    for mod in other_modules:
        device_map[mod] = n_gpus - 1

    # 2. Assign trainable blocks (0:upto_layer) evenly across GPUs
    trainable_blocks = block_names[:upto_layer]
    for idx, block in enumerate(trainable_blocks):
        device_map[block] = idx % n_gpus

    # 3. Assign remaining blocks (upto_layer:n_layers) evenly across GPUs
    non_trainable_blocks = block_names[upto_layer:]
    for idx, block in enumerate(non_trainable_blocks):
        device_map[block] = idx % n_gpus

    return device_map


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
        mt: ModelandTokenizer,
        regularization_dataloader: DataLoader = None,
        regularizer_lambda: float = 0.1,
        accelerator: Accelerator = None,
        initialize_trainable_params: bool = True,
    ):
        self.mt = mt
        self.regularization_dataloader = regularization_dataloader
        self.regularizer_lambda = regularizer_lambda
        self.cached_reg_info = None
        self.accelerator = accelerator
        if self.accelerator is None:
            self.accelerator = Accelerator()
        self.mt._model = self.accelerator.prepare(self.mt._model)
        self.regularization_dataloader = self.accelerator.prepare(
            self.regularization_dataloader
        )
        if self.regularization_dataloader is not None and self.regularizer_lambda > 0:
            self._cache_regularization_docs()

        self.trainable_params = None
        if initialize_trainable_params:
            self.initialize_trainable_params()

    @torch.inference_mode()
    def _cache_regularization_docs(self):
        """
        Cache regularization documents for later use during training.
        """
        self.cached_reg_info = []

        logger.info("Caching regularization documents...")
        for cur_batch in tqdm(self.regularization_dataloader):
            cur_batch = {k: v.to(self.mt.device) for k, v in cur_batch.items()}

            # #! Probably unfeasible to cache all the logits, will need to do it on the fly
            # with torch.no_grad():
            #     outputs = self.mt._model(
            #         input_ids=cur_batch["input_ids"],
            #         attention_mask=cur_batch["attention_mask"],
            #         labels=cur_batch["input_ids"],
            #     )

            #     batch_size = find_batch_size(cur_batch["input_ids"])
            #     loss = outputs.loss / batch_size

            self.cached_reg_info.append(
                {
                    "input_ids": cur_batch["input_ids"].detach().cpu(),
                    "attention_mask": cur_batch["attention_mask"].detach().cpu(),
                    # "loss": loss.detach().cpu(), #! will calculate loss on the fly
                }
            )

        # free_gpu_cache()
        logger.info(f"Cached {len(self.cached_reg_info)} regularization batches")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels=None,
        apply_modification=True,
    ):
        #! Not using `apply_param_delta`. Note that this is problamatic as now the logits for the
        #! regularization docs are being calcuated on the fly
        #! the regularization loss will always be zero.
        # TODO : fix this (maybe later)

        raise NotImplementedError("modify to use `apply_param_delta`")

        """
        Forward pass for the language model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            labels: Labels for the input (used for calculating loss)

        Returns:
            Loss value
        """
        input_ids = input_ids.to(self.mt.device)
        attention_mask = (
            attention_mask.to(self.mt.device) if attention_mask is not None else None
        )
        labels = labels.to(self.mt.device) if labels is not None else None

        with baukit.TraceDict(
            module=self.mt._model,
            retain_input=True,
            retain_output=True,
            retain_grad=True,
        ):
            output = self.mt._model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

        return output

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

        # Forward pass with the finetuning data.
        # apply usual next word prediction loss
        # logger.debug(
        #     f"STEP: applying next word prediction loss on {input_ids.shape = }"
        # )
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
            reg_input_ids = reg_doc["input_ids"].to(self.mt.device)
            reg_attention_mask = reg_doc["attention_mask"].to(self.mt.device)
            # orig_loss = reg_doc["loss"].to(self.model.device)

            # logger.debug(
            #     f"STEP: applying regularization loss on {reg_input_ids.shape = }"
            # )

            with torch.no_grad():
                orig_logits = self.forward(
                    input_ids=reg_input_ids,
                    attention_mask=reg_attention_mask,
                    apply_modification=False,
                ).logits

            # logger.debug(f"{orig_logits.shape=}")

            # Calculate current loss on regularization document
            reg_logits = self.forward(
                input_ids=reg_input_ids,
                attention_mask=reg_attention_mask,
                # labels=reg_input_ids,
                apply_modification=True,
            ).logits

            # logger.debug(f"{reg_logits.shape=}")

            # kldiv loss between the original logits and the regularized logits
            reg_loss = torch.nn.functional.kl_div(
                input=torch.nn.functional.log_softmax(reg_logits, dim=-1),
                target=torch.nn.functional.softmax(orig_logits, dim=-1),
                reduction="batchmean",
            )

            # print(f"{reg_loss=}")

            # divide by the sequence length
            reg_loss = reg_loss / reg_input_ids.shape[1]

            loss_dict["reg_loss"] = reg_loss.detach().item()

            # Combine losses
            loss += self.regularizer_lambda * reg_loss
            loss_dict["total_loss"] = loss.detach().item()

        # print("exiting loss function")
        return loss, loss_dict

    def train_mode(self):
        self.mt._model.train()

    def eval_mode(self):
        self.mt._model.eval()

    @torch.inference_mode()
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.mt._model)
        unwrapped_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def initialize_trainable_params(
        self,
        tunable_module_signatures=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    ):
        """
        Get the subset of model parameters to optimize.
        For LLaMA models, we only tune the parameters in the layers.
        """
        tunable_param_dict = {}
        for name, param in self.mt._model.named_parameters():
            if any(sig in name for sig in tunable_module_signatures):
                param.requires_grad = True
                module_name = ".".join(name.split(".")[:-1])
                assert (
                    module_name not in tunable_param_dict
                ), f"Module {module_name} already exists in tunable_param_dict"
                tunable_param_dict[module_name] = ParameterDelta(
                    module=get_module_nnsight(self.mt, module_name),
                    module_name=module_name,
                    param_delta=param,
                )

        # Calculate numbers for reporting
        trainable_params = sum(
            p.param_delta.numel() for p in tunable_param_dict.values()
        )

        self.trainable_params = tunable_param_dict

        if self.accelerator is not None:
            self.mt._model = self.accelerator.prepare(self.mt._model)

        # Log parameter counts
        logger.info(f"TRAINABLE PARAMS: {trainable_params / 1e9:.2f}B")

        return tunable_param_dict


class ParameterDelta(torch.nn.Module):
    def __init__(
        self,
        module: Envoy,
        module_name: str,
        param_delta: Optional[torch.nn.Parameter] = None,
    ):
        super().__init__()
        self.module = module
        self.module_name = module_name
        if param_delta is None:
            param = getattr(module, "weight")
            if param is None:
                raise ValueError(
                    f"Initialization Error, {module_name} does not have a weight"
                )
            self.param_delta = torch.nn.Parameter(
                torch.zeros_like(param).to(param.dtype).to(param.device)
            )
        else:
            self.param_delta = param_delta

        self.param_delta.requires_grad = True

    # ** nnsight specific implementation
    def __call__(self, inp: torch.Tensor):
        # h_delta = inp @ self.param_delta.t()
        # using torch implementation just to be safe
        h_delta = torch.nn.functional.linear(
            inp, self.param_delta, bias=None
        )  # (batch_size, seq_len, hidden_dim)

        return h_delta

    def parameters(self):
        return self.param_delta

    def __str__(self):
        return f"ParameterDelta(module={self.module}, param_name={self.module_name})"

    # # ** nnsight specific implementation
    # def apply_nnsight(self, context_manager=None, debug=False):
    #     """
    #     Apply the parameter delta to the module using nnsight.
    #     """
    #     if debug:
    #         if context_manager is None:
    #             logger.warning(
    #                 "Cannot log debug info without context manager. Setting debug to False"
    #             )
    #             debug = False

    #     if debug:
    #         context_manager.log(
    #             self.module_name, "param_delta shape: ", self.param_delta.shape
    #         )

    #     inp = self.module.input
    #     out = self.module.output

    #     if debug:
    #         context_manager.log(self.module_name, "inp shape: ", inp.shape)
    #         context_manager.log(self.module_name, "out shape: ", out.shape)
    #         context_manager.log(
    #             self.module_name, "param_delta shape: ", self.param_delta.shape
    #         )

    #     h_delta = self(inp)

    #     if debug:
    #         context_manager.log(self.module_name, "h_delta shape: ", h_delta.shape)

    #     # Apply the delta to the module's output
    #     self.module.output = out + h_delta

    @staticmethod
    def apply(trainable_params: dict[str, "ParameterDelta"]) -> callable:
        """
        Apply the parameter delta to the module.
        """

        def edit_repr(module_name: str, input: Any, output: Any):
            if module_name in trainable_params:
                #! delta intervention is not supported for layernorm yet
                if "layernorm" in module_name:
                    return output
                param_delta = trainable_params[module_name]
                # logger.debug(
                #     f"Applying param delta to {module_name} >> {param_delta.param_delta.shape=}"
                # )
            else:
                raise ValueError(f"Module {module_name} not found in param delta dict")

            input = untuple(input)

            output_0 = untuple(output)
            # logger.debug(f"input shape: {input.shape} | output shape: {output.shape}")

            h_delta = param_delta(input)
            # logger.debug(f"h_delta shape: {h_delta.shape}")

            output_0 += h_delta

            return output

        return edit_repr


class TrainableLM_delta(TrainableLM):
    def __init__(
        self,
        mt: ModelandTokenizer,
        regularization_dataloader: DataLoader = None,
        regularizer_lambda: float = 0.1,
        accelerator: Accelerator = None,
        trainable_params: Optional[str | torch.nn.ModuleDict] = None,
        block_indices: Optional[list[int]] = None,
        tunable_module_signatures: Optional[List[str]] = [
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    ):
        super().__init__(
            mt=mt,
            regularization_dataloader=regularization_dataloader,
            regularizer_lambda=regularizer_lambda,
            accelerator=accelerator,
            initialize_trainable_params=False,
        )
        self.block_indices = (
            block_indices if block_indices is not None else list(range(self.mt.n_layer))
        )
        self.tunable_module_signatures = tunable_module_signatures
        if trainable_params is not None:
            self.load_trainable_params(trainable_params)
        else:
            self.initialize_trainable_params(
                tunable_module_signatures=self.tunable_module_signatures
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels=None,
        apply_modification=True,
    ):
        """
        Forward pass for the language model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            labels: Labels for the input (used for calculating loss)

        Returns:
            Loss value
        """
        input_ids = input_ids.to(self.mt.device)
        attention_mask = (
            attention_mask.to(self.mt.device) if attention_mask is not None else None
        )
        labels = labels.to(self.mt.device) if labels is not None else None

        with baukit.TraceDict(
            module=self.mt._model,
            layers=list(self.trainable_params.keys()),
            retain_input=True,
            retain_output=True,
            retain_grad=True,
            edit_output=(
                ParameterDelta.apply(self.trainable_params)
                if apply_modification
                else None
            ),
        ):
            output = self.mt._model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

        return output

    @torch.inference_mode()
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        trainable_params = {
            name.replace(".", "<>"): param_delta.parameters()
            for name, param_delta in self.trainable_params.items()
        }
        torch.save(trainable_params, os.path.join(path, "trainable_params.pt"))
        logger.info(f"trainable_params saved to {path}")

    def initialize_trainable_params(
        self,
        tunable_module_signatures=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    ):
        """
        Get the subset of model parameters to optimize.
        For LLaMA models, we only tune the parameters in the layers.
        """
        for param in self.mt._model.parameters():
            param.requires_grad = False

        tunable_param_dict = {}
        for block_idx in self.block_indices:
            block_name = self.mt.layer_name_format.format(block_idx)
            block = baukit.get_module(self.mt._model, block_name)
            for name, param in block.named_parameters():
                if (
                    tunable_module_signatures is not None
                    and any(sig in name for sig in tunable_module_signatures) is False
                ):
                    continue
                with torch.no_grad():
                    param_delta = (
                        torch.nn.Parameter(
                            torch.zeros_like(param).to(param.dtype).to(param.device)
                        )
                        # + 5 # only for testing if the param_delta is being applied
                    )
                param_delta.requires_grad = True
                param_delta = self.accelerator.prepare(param_delta)
                module_name = block_name + "." + ".".join(name.split(".")[:-1])
                logger.debug(f"{module_name=}")
                assert (
                    module_name not in tunable_param_dict
                ), f"Module {module_name} already exists in tunable_param_dict"
                tunable_param_dict[module_name] = ParameterDelta(
                    module=get_module_nnsight(self.mt, module_name),
                    module_name=module_name,
                    param_delta=param_delta,
                )

        # for name, param in self.mt._model.named_parameters():
        #     if any(sig in name for sig in tunable_module_signatures):
        #         with torch.no_grad():
        #             param_delta = (
        #                 torch.nn.Parameter(
        #                     torch.zeros_like(param).to(param.dtype).to(param.device)
        #                 )
        #                 # + 5 # only for testing if the param_delta is being applied
        #             )
        #         param_delta.requires_grad = True
        #         param_delta = self.accelerator.prepare(param_delta)
        #         module_name = ".".join(name.split(".")[:-1])
        #         assert (
        #             module_name not in tunable_param_dict
        #         ), f"Module {module_name} already exists in tunable_param_dict"
        #         tunable_param_dict[module_name] = ParameterDelta(
        #             module=get_module_nnsight(self.mt, module_name),
        #             module_name=module_name,
        #             param_delta=param_delta,
        #         )

        # Calculate numbers for reporting
        trainable_params = sum(
            p.param_delta.numel() for p in tunable_param_dict.values()
        )

        self.trainable_params = tunable_param_dict

        if self.accelerator is not None:
            self.mt._model = self.accelerator.prepare(self.mt._model)

        # Log parameter counts
        logger.info(f"TRAINABLE PARAMS: {trainable_params / 1e9:.2f}B")

        return tunable_param_dict

    def load_trainable_params(self, trainable_param_dict: str | torch.nn.ModuleDict):
        """
        Load the parameter delta dictionary from a file or a module.
        """
        for param in self.mt._model.parameters():
            param.requires_grad = False

        if isinstance(trainable_param_dict, str):
            trainable_param_dict = torch.load(trainable_param_dict)

        self.trainable_params = {}
        for module_name, param in trainable_param_dict.items():
            module_name = module_name.replace("<>", ".")
            base_module = get_module_nnsight(self.mt, module_name)
            base_params = getattr(base_module, "weight")
            self.trainable_params[module_name] = ParameterDelta(
                module=base_module,
                module_name=module_name,
                param_delta=param.to(base_params.dtype).to(base_params.device),
            )

        # Calculate numbers for reporting
        trainable_params = sum(
            p.param_delta.numel() for p in self.trainable_params.values()
        )

        # Log parameter counts
        logger.info(f"TRAINABLE PARAMS: {trainable_params / 1e9:.2f}B")

    def _get_tunable_params(self):
        trainable_params = [
            param.param_delta for param in self.trainable_params.values()
        ]
        return trainable_params

    def train_mode(self):
        self.mt._model.train()

    def eval_mode(self):
        self.mt._model.eval()

    def apply_clamp(self, clamp_value: float = 1e-5):
        """
        Clamp the absolute value of the parameter deltas to a maximum value.
        """
        for module_name, param_delta in self.trainable_params.items():
            # logger.debug(f"clamping {module_name} | {param_delta.param_delta.data.shape=}")
            with torch.no_grad():
                param_delta.param_delta.data = torch.clamp(
                    param_delta.param_delta.data,
                    min=-clamp_value,
                    max=clamp_value,
                )

    @staticmethod
    def fuse_with_model(model: Model, param_delta_dict: torch.nn.ModuleDict):
        for module_name, param_delta in param_delta_dict.items():
            module_name = module_name.replace("<>", ".")
            logger.debug(f"{module_name=} | {param_delta.shape=}")
            module = baukit.get_module(model, module_name)
            with torch.no_grad():
                module.weight[...] = module.weight + param_delta.to(
                    module.weight.dtype
                ).to(module.weight.device)

    @staticmethod
    def defuse_from_model(model: Model, param_delta_dict: torch.nn.ModuleDict):
        for module_name, param_delta in param_delta_dict.items():
            module_name = module_name.replace("<>", ".")
            logger.debug(f"{module_name=} | {param_delta.shape=}")
            module = baukit.get_module(model, module_name)
            with torch.no_grad():
                module.weight[...] = module.weight - param_delta.to(
                    module.weight.dtype
                ).to(module.weight.device)


class ParameterLoRA(torch.nn.Module):
    def __init__(
        self,
        module: Envoy,
        module_name,
        W_left: Optional[torch.nn.Parameter] = None,
        W_right: Optional[torch.nn.Parameter] = None,
        rank: int = 128,
        init_scale: float = 0.01,  # Scale factor for initialization
    ):
        super().__init__()
        self.module = module
        self.module_name = module_name
        if W_left is not None or W_right is not None:
            assert (
                W_left is not None and W_right is not None
            ), "Both W_left and W_right should be provided for LORA"
            self.W_left = W_left
            self.W_right = W_right
        else:
            param = getattr(module, "weight")
            if param is None:
                raise ValueError(
                    f"Initialization Error, {module_name} does not have a weight"
                )
            inp_dim = param.shape[0]
            out_dim = param.shape[1]

            # Use better initialization techniques:
            # 1. Kaiming/He initialization for W_left (scaled)
            self.W_left = torch.nn.Parameter(
                torch.nn.init.kaiming_normal_(
                    torch.empty((inp_dim, rank)), a=math.sqrt(5)
                )
                .to(param.dtype)
                .to(param.device)
                * init_scale
            )

            # 2. Zero initialization for W_right (common practice in LoRA papers)
            # This creates a "near-identity" initial behavior where the LoRA contribution starts small
            self.W_right = torch.nn.Parameter(
                torch.zeros((rank, out_dim)).to(param.dtype).to(param.device)
            )

            # logger.debug(
            #     f"{param.shape=} | {self.W_left.shape=} | {self.W_right.shape=}"
            # )

        self.W_left.requires_grad = True
        self.W_right.requires_grad = True

    def __call__(self, inp: torch.Tensor):
        # logger.debug(
        #     f"{self.module_name} | {inp.shape=} | {self.W_left.shape=} | {self.W_right.shape=}"
        # )
        # h_delta = (inp @ self.W_right.t()) @ self.W_left.t()
        h_intermediate = torch.nn.functional.linear(inp, self.W_right)
        h_delta = torch.nn.functional.linear(h_intermediate, self.W_left)
        return h_delta

    def parameters(self):
        return [self.W_left, self.W_right]

    def numel(self):
        """Return the total number of parameters"""
        return self.W_left.numel() + self.W_right.numel()

    def __str__(self):
        return f"ParameterLORA(module={self.module}, param_name={self.module_name})"

    @staticmethod
    def apply(trainable_params: dict[str, "ParameterLoRA"]) -> callable:
        """
        Apply the LoRA parameter modification to the module.
        """

        def edit_repr(module_name: str, input: Any, output: Any):
            if module_name in trainable_params:
                param_lora = trainable_params[module_name]
            else:
                raise ValueError(f"Module {module_name} not found in param LoRA dict")

            input = untuple(input)
            output_0 = untuple(output)

            h_delta = param_lora(input)
            output_0 += h_delta

            return output

        return edit_repr


class TrainableLM_LoRA(TrainableLM):
    def __init__(
        self,
        mt: ModelandTokenizer,
        regularization_dataloader: DataLoader = None,
        regularizer_lambda: float = 0.1,
        accelerator: Accelerator = None,
        trainable_params: Optional[str | torch.nn.ModuleDict] = None,
        rank: int = 128,
    ):
        super().__init__(
            mt=mt,
            regularization_dataloader=regularization_dataloader,
            regularizer_lambda=regularizer_lambda,
            accelerator=accelerator,
            initialize_trainable_params=False,
        )

        self.rank = rank
        if trainable_params is not None:
            self.load_trainable_params(trainable_params)
        else:
            self.initialize_trainable_params()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels=None,
        apply_modification=True,
    ):
        """
        Forward pass for the language model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            labels: Labels for the input (used for calculating loss)

        Returns:
            Loss value
        """
        input_ids = input_ids.to(self.mt.device)
        attention_mask = (
            attention_mask.to(self.mt.device) if attention_mask is not None else None
        )
        labels = labels.to(self.mt.device) if labels is not None else None

        with baukit.TraceDict(
            module=self.mt._model,
            layers=list(self.trainable_params.keys()),
            retain_input=True,
            retain_output=True,
            retain_grad=True,
            edit_output=(
                ParameterLoRA.apply(self.trainable_params)
                if apply_modification
                else None
            ),
        ):
            output = self.mt._model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

        return output

    @torch.inference_mode()
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        trainable_params = {}
        for name, param_lora in self.trainable_params.items():
            trainable_params[name.replace(".", "<>")] = {
                "W_left": param_lora.W_left,
                "W_right": param_lora.W_right,
            }
        torch.save(trainable_params, os.path.join(path, "trainable_params_lora.pt"))
        logger.info(f"trainable_params saved to {path}")

    def initialize_trainable_params(
        self,
        tunable_module_signatures=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    ):
        """
        Initialize LoRA parameters for the specified modules.
        For LLaMA models, we typically tune the parameters in the MLP layers.
        """
        tunable_param_dict = {}
        for name, param in self.mt._model.named_parameters():
            if any(sig in name for sig in tunable_module_signatures):
                module_name = ".".join(name.split(".")[:-1])
                if module_name in tunable_param_dict:
                    continue  # Skip if we already added this module

                module = get_module_nnsight(self.mt, module_name)
                param_lora = ParameterLoRA(
                    module=module,
                    module_name=module_name,
                    rank=self.rank,
                )

                # Prepare with accelerator
                param_lora.W_left = self.accelerator.prepare(param_lora.W_left)
                param_lora.W_right = self.accelerator.prepare(param_lora.W_right)

                tunable_param_dict[module_name] = param_lora

        # Calculate numbers for reporting
        trainable_params = sum(p.numel() for p in tunable_param_dict.values())

        self.trainable_params = tunable_param_dict

        if self.accelerator is not None:
            self.mt._model = self.accelerator.prepare(self.mt._model)

        # Log parameter counts
        logger.info(f"TRAINABLE PARAMS: {trainable_params / 1e9:.2f}B")
        logger.info(f"Using LoRA with rank {self.rank}")

        return tunable_param_dict

    def load_trainable_params(self, trainable_param_dict: str | torch.nn.ModuleDict):
        """
        Load the LoRA parameters from a file or a module.
        """
        if isinstance(trainable_param_dict, str):
            trainable_param_dict = torch.load(trainable_param_dict)

        self.trainable_params = {}
        for module_name, param in trainable_param_dict.items():
            module_name = module_name.replace("<>", ".")
            base_module = get_module_nnsight(self.mt, module_name)
            base_params = getattr(base_module, "weight")

            W_left = param["W_left"].to(base_params.dtype).to(base_params.device)
            W_right = param["W_right"].to(base_params.dtype).to(base_params.device)

            self.trainable_params[module_name] = ParameterLoRA(
                module=base_module,
                module_name=module_name,
                W_left=W_left,
                W_right=W_right,
            )

        # Calculate numbers for reporting
        trainable_params = sum(p.numel() for p in self.trainable_params.values())

        # Infer rank from W_left dimension
        self.rank = list(self.trainable_params.values())[0].W_left.shape[1]

        # Log parameter counts
        logger.info(f"TRAINABLE PARAMS: {trainable_params / 1e9:.2f}B")
        logger.info(f"Using LoRA with rank {self.rank}")

    def _get_tunable_params(self):
        """
        Get all trainable parameters from the LoRA modules.
        Returns a flat list of parameters for the optimizer.
        """
        params = []
        for param_lora in self.trainable_params.values():
            params.extend(param_lora.parameters())
        return params

    @staticmethod
    def fuse_with_model(model: Model, param_lora_dict: torch.nn.ModuleDict):
        """
        Fuse the LoRA parameters with the base model.
        This modifies the model in place.
        """
        for module_name, param_lora in param_lora_dict.items():
            module_name = module_name.replace("<>", ".")
            module = baukit.get_module(model, module_name)
            # logger.debug(
            #     f'{module_name=} | {module.weight.shape} | {param_lora["W_left"].shape=} | {param_lora["W_right"].shape=}'
            # )
            with torch.no_grad():
                module.weight[...] = (
                    module.weight + param_lora["W_left"] @ param_lora["W_right"]
                )
        logger.info("Fused LoRA parameters with the model")

    @staticmethod
    def defuse_from_model(model: Model, param_lora_dict: torch.nn.ModuleDict):
        """
        Defuse the LoRA parameters from the base model.
        This modifies the model in place.
        """
        for module_name, param_lora in param_lora_dict.items():
            module_name = module_name.replace("<>", ".")
            # logger.debug(
            #     f"{module_name=} | {param_lora.W_left.shape=} | {param_lora.W_right.shape=}"
            # )
            module = baukit.get_module(model, module_name)
            with torch.no_grad():
                module.weight[...] = (
                    module.weight - param_lora.W_left @ param_lora.W_right.t()
                )
        logger.info("Defused LoRA parameters from the model")


###########################################################  TRAINER  ###########################################################


# * A Trainer class inspired by the design of Pytorch Lightning (which doesn't work for accelerate, hence the need for this class)
# TODO(arnab) Does not support adding custom callbacks yet.
# TODO?(arnab) Train currently does not work for nnsight. Carrying around nnsight context manager is awkward and probably not worth it (at this moment)
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
        weight_decay: float = 1e-3,
        clamp_abs_update: Optional[float] = None,
        warmup_steps: int = 50,
        # checkpointing
        save_path: str = "ft_checkpoints",
        save_interval: int = 10,
        keep_checkpoints: List[int] = None,
        remove_old_checkpoints: bool = True,
        # memory management
        memory_cleaner_threshold: float = 0.7,
        # wandb logging
        log_to_wandb: bool = True,
        optimizer_function: callable = AdamW,
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
        self.clamp_abs_update = clamp_abs_update

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
        self.accelerator = (
            Accelerator() if trainable.accelerator is None else trainable.accelerator
        )

        # Create optimizer and scheduler
        self._setup_optimizer_and_scheduler(optimizer_function=optimizer_function)

        # Prepare model and dataloaders with accelerator
        (
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

    @property
    def hparams(self) -> dict[str, Any]:
        """
        Get hyperparameters for the trainer.

        Returns:
            Dictionary of hyperparameters
        """
        return {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "save_path": self.save_path,
            "save_interval": self.save_interval,
            "keep_checkpoints": self.keep_checkpoints,
            "remove_old_checkpoints": self.remove_old_checkpoints,
            "memory_cleaner_threshold": self.memory_cleaner_threshold,
            "log_to_wandb": self.log_to_wandb,
            "clamp_abs_update": self.clamp_abs_update,
        }

    def _setup_optimizer_and_scheduler(self, optimizer_function: callable = AdamW):
        """Set up optimizer and learning rate scheduler."""
        # Get tunable parameters
        tunable_params = self.trainable._get_tunable_params()

        # Create optimizer with proper configuration for 8-bit optimizers
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.95),
        }

        # For bitsandbytes 8-bit optimizers, add specific configurations
        if (
            hasattr(optimizer_function, "__module__")
            and "bitsandbytes" in optimizer_function.__module__
        ):
            # Remove betas for bitsandbytes optimizers as they have different parameter names
            optimizer_kwargs.pop("betas", None)
            # Add bitsandbytes-specific parameters
            optimizer_kwargs.update(
                {
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "optim_bits": 8,  # Ensure 8-bit optimization
                    "percentile_clipping": 100,  # Optional: can help with stability
                }
            )
            logger.info("Configuring 8-bit optimizer with specialized parameters")

        self.optimizer = optimizer_function(tunable_params, **optimizer_kwargs)

        # Calculate total number of training steps
        total_steps = (
            len(self.train_dataloader)
            * self.num_epochs
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

        total_steps = max(total_steps, 100000)

        logger.info(f"Setting total training steps: {total_steps}")

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

        # Save model
        self.trainable.save(save_dir)

    def train(self):
        """
        Train the model for the specified number of epochs.

        Args:
            num_epochs: Number of epochs to train for
        """
        # Log the total number of epochs
        logger.info(f"Starting training for {self.num_epochs} epochs")

        # Run the initial evaluation
        eval_results = self.evaluate()

        # Log epoch-level metrics directly to wandb
        if self.log_to_wandb and self.accelerator.is_local_main_process:
            wandb_epoch_report = {"epoch": 0}
            wandb_epoch_report["epoch/val_loss"] = eval_results["loss"]
            wandb_epoch_report["epoch/val_perplexity"] = eval_results["perplexity"]
            logger.info("Logging epoch-level metrics to wandb", wandb_epoch_report)
            wandb.log(wandb_epoch_report)

        log_optim_overhead = True
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Initial GPU memory usage: {initial_memory:.2f} GB")

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
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            # Batch loop
            for batch_idx, batch in enumerate(progress_bar):
                # print(f"{batch_idx=}")
                # print(batch)

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

                # Log memory usage after optimizer preparation
                if batch_idx == 5 and log_optim_overhead and torch.cuda.is_available():
                    log_optim_overhead = False
                    post_optimizer_memory = torch.cuda.memory_allocated() / 1e9
                    optimizer_memory_overhead = post_optimizer_memory - initial_memory
                    logger.info(
                        f"Memory after optimizer setup: {post_optimizer_memory:.2f} GB"
                    )
                    logger.info(
                        f"Optimizer memory overhead: {optimizer_memory_overhead:.2f} GB"
                    )

                self.optimizer.zero_grad()

                # Clip gradients if clamp_abs_update is set
                if self.clamp_abs_update is not None:
                    assert type(self.trainable) is TrainableLM_delta
                    self.trainable.apply_clamp(self.clamp_abs_update)

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
                # Update progress bar
                progress_bar.set_postfix(
                    {k: v / (batch_idx + 1) for k, v in total_loss_dict.items()}
                )

                # # Maybe clean up memory
                # if batch_idx % 10 == 0:
                #     self._maybe_cleanup_memory()

            for k in total_loss_dict:
                total_loss_dict[k] /= num_batches

            # Log epoch metrics
            loss_log = ""
            for k, v in total_loss_dict.items():
                loss_log += f"{k}: {v:.4f} | "
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} | {loss_log}")

            free_gpu_cache()

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
            if epoch + 1 < self.num_epochs:  # no need to save the final model twice
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
