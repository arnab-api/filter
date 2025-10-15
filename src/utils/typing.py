"""Some useful type aliases relevant to this project."""

import pathlib
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy
import torch
import transformers
import transformers.modeling_outputs
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel

ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = str | pathlib.Path
Device = str | torch.device

# Throughout this codebase, we use HuggingFace model implementations.
Model = (
    LanguageModel
    | transformers.GPT2LMHeadModel
    | transformers.GPTJForCausalLM
    | transformers.GPTNeoXForCausalLM
    | transformers.LlamaForCausalLM
    | transformers.Gemma3ForConditionalGeneration
    | transformers.Gemma2ForCausalLM
    | transformers.GemmaForCausalLM
    | transformers.Qwen2ForCausalLM
    | transformers.Olmo2ForCausalLM
    | transformers.OlmoForCausalLM
    | transformers.Qwen3ForCausalLM
)
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping = Sequence[tuple[int, int]]
TokenizerOutput = transformers.tokenization_utils_base.BatchEncoding

ModelInput = transformers.BatchEncoding
ModelOutput = transformers.modeling_outputs.CausalLMOutput
ModelGenerateOutput = transformers.generation.utils.GenerateOutput | torch.LongTensor

Layer = int | Literal["emb"] | Literal["ln_f"]

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = list[str] | tuple[str, ...]


@dataclass(frozen=False)
class PredictedToken(DataClassJsonMixin):
    """A predicted token and its probability."""

    token: str
    prob: Optional[float] = None
    logit: Optional[float] = None
    token_id: Optional[int] = None
    metadata: Optional[dict] = None

    def __str__(self) -> str:
        rep = f'"{self.token}"[{self.token_id}]'
        # if self.prob is not None:
        #     rep += f" (p={self.prob:.3f})"
        # if self.logit is not None:
        #     rep += f" (logit={self.logit:.3f})"
        add = []
        if self.prob is not None:
            add.append(f"p={self.prob:.3f}")
        if self.logit is not None:
            add.append(f"logit={self.logit:.3f}")

        if self.metadata is not None:
            for k, v in self.metadata.items():
                if k not in ["token", "token_id"] and v is not None:
                    add.append(f"{k}={v}")

        if len(add) > 0:
            rep += " (" + ", ".join(add) + ")"
        return rep


@dataclass(frozen=False, kw_only=True)
class SVD:
    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor

    def __post_init__(self):
        # assert self.U.shape[1] == self.S.shape[0]
        # assert self.S.shape[0] == self.Vh.shape[1]
        # assert self.V.shape[0] == self.V.shape[1], "Vh must be square"
        pass

    @property
    def shape(self):
        return self.U.shape, self.S.shape, self.V.shape

    def to_device(self, device: torch.device):
        """in-place device change -- to save memory"""
        self.U = self.U.to(device)
        self.S = self.S.to(device)
        self.V = self.V.to(device)
        return self

    def to_dtype(self, dtype: torch.dtype):
        """in-place dtype change -- to save memory"""
        self.U = self.U.to(dtype)
        self.S = self.S.to(dtype)
        self.V = self.V.to(dtype)
        return self

    @property
    def dtype(self):
        return self.U.dtype

    @property
    def device(self):
        return self.U.device

    @staticmethod
    def calculate(matrix: torch.Tensor) -> "SVD":
        n, d = matrix.shape
        if n >= d:
            # U, S, V = torch.svd(matrix.to(dtype=torch.float32))
            U, S, Vh = torch.linalg.svd(
                matrix.to(dtype=torch.float32), full_matrices=False
            )
            V = Vh.mH
        else:
            U, S, Vh = torch.linalg.svd(
                matrix.to(dtype=torch.float32), full_matrices=True
            )
            V = Vh.mH

        svd = SVD(U=U, S=S, V=V)
        svd.to_dtype(matrix.dtype)
        svd.to_device(matrix.device)
        return svd
