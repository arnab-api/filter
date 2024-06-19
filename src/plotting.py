import os
from typing import Literal, Optional

import matplotlib.pyplot as plt
import torch

from src.trace import CausalTracingResult


def get_color_map(kind: Literal["residual", "mlp", "attention"] = "residual"):
    if kind == "residual":
        return "Purples"
    if kind == "mlp":
        return "Greens"
    if kind == "attention":
        return "Reds"
    return "Greys"


def replace_special_tokens(token_list, pad_token="[PAD]"):
    for i, token in enumerate(token_list):
        if token.startswith("<|") and token.endswith("|>"):
            token_list[i] = pad_token
    return token_list


def plot_trace_heatmap(
    result: CausalTracingResult,
    savepdf: Optional[str] = None,
    model_name: Optional[str] = None,
    scale_range: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
):
    scores = result.indirect_effects
    clean_tokens = replace_special_tokens(result.clean_input_toks)
    corrupt_tokens = replace_special_tokens(result.corrupt_input_toks)
    tokens = clean_tokens[result.trace_start_idx :]
    for t_idx in range(result.subj_end - result.trace_start_idx):
        tokens[t_idx] = (
            f"{clean_tokens[t_idx + result.trace_start_idx]}/{corrupt_tokens[t_idx + result.trace_start_idx]}"
        )

    with plt.rc_context(
        rc={
            "font.family": "Times New Roman",
            "font.size": 10,
        }
    ):
        fig, ax = plt.subplots(figsize=(3.5, len(tokens) * 0.08 + 1.8), dpi=200)
        scale_kwargs = dict(
            vmin=result.low_score if scale_range is None else scale_range[0],
        )
        if scale_range is not None:
            scale_kwargs["vmax"] = scale_range[1]

        heatmap = ax.pcolor(
            scores,
            cmap=get_color_map(result.kind),
            **scale_kwargs,
        )

        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(scores))])
        ax.set_xticks([0.5 + i for i in range(0, scores.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, scores.shape[1] - 6, 5)))
        ax.set_yticklabels(tokens)

        if title is None:
            title = f"Indirect Effects of {result.kind.upper()} Layers"
        ax.set_title(title)

        if result.window == 1:
            ax.set_xlabel(f"single restored layer within {model_name}")
        else:
            ax.set_xlabel(
                f"center of interval of {result.window} restored {result.kind.upper()} layers"
            )

        color_scale = plt.colorbar(heatmap)
        color_scale.ax.set_title(
            f"p({result.answer.token.strip()})", y=-0.12, fontsize=10
        )

        if savepdf is not None:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight", dpi=300)
        plt.show()


import matplotlib.pyplot as plt


def visualize_attn_matrix(
    attn_matrix: torch.Tensor,
    tokens: list[str],
    remove_eos: Optional[str] = None,
    title: str | None = None,
    color_scheme: str = "Blues",
):
    assert (
        attn_matrix.shape[0] == attn_matrix.shape[1]
    ), "Attention matrix must be square"
    assert (
        len(tokens) == attn_matrix.shape[-1]
    ), "Tokens and attention matrix must have the same length"

    start_idx = 0
    if remove_eos:
        start_idx = 1 if tokens[0] == remove_eos else 0

    with plt.rc_context(
        rc={
            "font.family": "Times New Roman",
            # "font.size": 13,
        }
    ):

        img = plt.imshow(
            attn_matrix[start_idx:, start_idx:],
            cmap=color_scheme,
            interpolation="nearest",
        )
        plt.colorbar(img, orientation="vertical")

        plt.xticks(
            range(len(tokens) - start_idx),
            [f'" {t}"' for t in tokens[start_idx:]],
            rotation=90,
        )
        plt.yticks(
            range(len(tokens) - start_idx),
            [f'" {t}"' for t in tokens[start_idx:]],
        )

        plt.ylabel("Query Token")
        plt.xlabel("Key Token")

        if title:
            plt.title(title)

        plt.show()
