from typing import Any, Literal

import torch
from nnsight import LanguageModel

from src.models import ModelandTokenizer
from src.utils.typing import Tokenizer


@torch.inference_mode()
def interpret_logits(
    tokenizer: ModelandTokenizer | Tokenizer,
    logits: torch.Tensor,
    k: int = 10,
    get_proba: bool = False,
) -> list[tuple[str, float]]:
    tokenizer = unwrap_tokenizer(tokenizer)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    token_ids = logits.topk(dim=-1, k=k).indices.squeeze().tolist()
    logit_values = logits.topk(dim=-1, k=k).values.squeeze().tolist()
    return [(tokenizer.decode(t), round(v, 3)) for t, v in zip(token_ids, logit_values)]


@torch.inference_mode()
def logit_lens(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    after_layer_norm: bool = False,
    interested_tokens: list[int] = [],
    get_proba: bool = False,
    k: int = 10,
) -> tuple[list[tuple[str, float]], dict]:
    lm_head = mt.lm_head if not after_layer_norm else mt.lm_head.lm_head
    h = untuple(h) if after_layer_norm else h
    logits = lm_head(h)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    # don't pass `get_proba` or softmax will be applied twice with `get_proba=True`
    candidates = interpret_logits(mt, logits, k=k)
    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        interested_logits = {
            t: {
                "p": logits[t].item(),
                "rank": rank_tokens.index(t) + 1,
                "token": mt.tokenizer.decode(t),
            }
            for t in interested_tokens
        }
        return candidates, interested_logits
    return candidates


def untuple(object: Any):
    if isinstance(object, tuple) or "LanguageModelProxy" in str(type(object)):
        return object[0]
    return object


def unwrap_model(mt: ModelandTokenizer | torch.nn.Module) -> torch.nn.Module:
    if isinstance(mt, ModelandTokenizer):
        return mt.model
    if isinstance(mt, torch.nn.Module):
        return mt
    raise ValueError("mt must be a ModelandTokenizer or a torch.nn.Module")


def unwrap_tokenizer(mt: ModelandTokenizer | Tokenizer) -> Tokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt


# useful for logging
def bytes_to_human_readable(
    size: int, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return f"{size / denom:.3f} {unit}"
