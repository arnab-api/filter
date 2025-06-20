import copy
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from src.functional import (
    find_token_range,
    free_gpu_cache,
    get_module_nnsight,
    interpret_logits,
    low_rank_pinv,
    prepare_input,
)
from src.models import ModelandTokenizer
from src.utils.typing import SVD, TokenizerOutput

logger = logging.getLogger(__name__)


@torch.inference_mode()
def project_to_vocab(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    layer_name: str,
    inputs: Optional[TokenizerOutput] = None,
    placeholder_pos: int = 0,
    **kwargs,
):
    if inputs is None:
        inputs = mt.tokenizer(
            mt.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
        )
        placeholder_pos = 0

    with mt.trace(inputs) as tr:
        module = get_module_nnsight(mt, layer_name)
        current_state = (
            module.output.save()
            if ("mlp" in layer_name or layer_name == mt.embedder_name)
            else module.output[0].save()
        )
        current_state[:, placeholder_pos, :] = h
        logits = mt.output.logits[0, -1].save()

    free_gpu_cache()

    return interpret_logits(
        tokenizer=mt,
        logits=logits,
        **kwargs,
    )


def module_output_has_extra_dim(mt, module_name):
    return (
        "mlp" not in module_name
        or module_name != mt.embedder_name
        or module_name != mt.lm_head_name
    )


def get_lm_head_row(mt: ModelandTokenizer, token: int):
    lm_head = get_module_nnsight(mt, "lm_head")
    return lm_head.weight[token].squeeze()


@dataclass(frozen=False, kw_only=True)
class Order1Approx:
    jacobian: torch.Tensor
    bias: torch.Tensor
    calculated_at: torch.Tensor
    output: torch.Tensor

    inp_layer: str
    out_layer: str = "lm_head"

    beta: float = 1.0  # scaling factor / make the slope steeper

    _svd: Optional[SVD] = None

    def __post_init__(self):
        # loaded in float16 to save memory
        self.to_dtype(torch.float16)

    @torch.inference_mode()
    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        return (
            self.beta * self.jacobian @ h.to(self.dtype).to(self.device) + self.bias
        ).to(h.dtype)

    @staticmethod
    def load_from_npz(path: str):
        data = np.load(path)
        return Order1Approx(
            jacobian=torch.Tensor(data["jacobian"]),
            bias=torch.Tensor(data["bias"]),
            calculated_at=torch.tensor(data["calculated_at"]),
            inp_layer=data["inp_layer"].item(),
            out_layer=data["out_layer"].item(),
            beta=data["beta"].item(),
        )

    def to_device(self, device: torch.device):
        self.jacobian = self.jacobian.to(device)
        self.bias = self.bias.to(device)
        self.calculated_at = self.calculated_at.to(device)
        if self._svd is not None:
            self._svd = self._svd.to_device(device)
        return self

    def to_dtype(self, dtype: torch.dtype):
        self.jacobian = self.jacobian.to(dtype)
        self.bias = self.bias.to(dtype)
        self.calculated_at = self.calculated_at.to(dtype)
        if self._svd is not None:
            self._svd = self._svd.to_dtype(dtype)
        return self

    @property
    def dtype(self):
        return self.jacobian.dtype

    @property
    def device(self):
        return self.jacobian.device

    def jacobian_inv(self, rank: int = 1000):
        if self._svd is None:
            self._svd = SVD.calculate(self.jacobian.float())
            self._svd.to_dtype(self.dtype).to_device(self.device)

        return low_rank_pinv(
            matrix=self.jacobian,
            svd=self._svd,
            rank=rank,
        )


def patch(
    h: torch.Tensor,
    mt: ModelandTokenizer,
    inp_layer: str,
    out_layer: str = "lm_head",
    context: TokenizerOutput | None = None,
    h_idx: int = 0,
    z_idx: int = -1,  # usually the last token
) -> torch.Tensor:
    if context is None:
        context = mt.tokenizer(
            mt.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
        )
        if h_idx != 0:
            logger.warning(
                "Context not provided. Using BOS token as context. Setting h_idx to 0."
            )
        h_idx = 0

    with mt.trace(context) as tr:
        # perform the patching
        inp_module = get_module_nnsight(mt, inp_layer)
        inp_state = (
            inp_module.output[0].save()
            if module_output_has_extra_dim(mt, inp_layer)
            else inp_module.output.save()
        )
        inp_state[:, h_idx, :] = h
        # tr.log(inp_state.shape)

        out_module = get_module_nnsight(mt, out_layer)
        out_state = (
            out_module.output[0].save()
            if module_output_has_extra_dim(mt, out_layer)
            else out_module.output.save()
        )

    # print(f"{inp_state[:, h_idx, :].norm().item()=}")
    return out_state[:, z_idx].squeeze()


def get_inputs_and_intervention_range(
    mt: ModelandTokenizer,
    prompt: str,
    intervention_token: str,
):
    inputs = prepare_input(
        prompts=[prompt],
        tokenizer=mt,
        return_offsets_mapping=True,
        device=mt.device,
    )

    offset_mapping = inputs["offset_mapping"][0]
    inputs.pop("offset_mapping")

    subj_range = find_token_range(
        string=prompt,
        substring=intervention_token,
        tokenizer=mt,
        offset_mapping=offset_mapping,
    )

    return inputs, subj_range


def order_1_approx(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    inp_layer: str,
    out_layer: str | None = None,
) -> Order1Approx:
    def func(h: torch.Tensor) -> torch.Tensor:
        return patch(h, mt, inp_layer, out_layer)

    def calculate_jacobian(function, h):
        h = h.to(mt.device)
        h.requires_grad = True
        h.retain_grad()
        z_est = function(h)
        jacobian = []

        for idx in tqdm(range(z_est.shape[0])):
            mt._model.zero_grad()
            z_est[idx].backward(retain_graph=True)
            jacobian.append(copy.deepcopy(h.grad))
            h.grad.zero_()

        jacobian = torch.stack(jacobian)

        return jacobian

    out_layer = mt.layer_names[-1] if out_layer is None else out_layer
    z = func(h)
    # jacobian = torch.autograd.functional.jacobian(func, h, vectorize=False)
    jacobian = calculate_jacobian(func, h)
    bias = z - jacobian @ h

    mt._model.zero_grad()
    free_gpu_cache()

    return Order1Approx(
        jacobian=jacobian,
        bias=bias,
        calculated_at=h,
        inp_layer=inp_layer,
        out_layer=out_layer,
    )
