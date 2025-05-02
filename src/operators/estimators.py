import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.functional import free_gpu_cache, get_module_nnsight
from src.models import ModelandTokenizer
from src.tokens import find_token_range, prepare_input

logger = logging.getLogger(__name__)


from src.operators.operators import Basis, CornerOperator
from src.operators.utils import project_to_vocab


@dataclass(frozen=False, kw_only=True)
class Estimator:
    mt: ModelandTokenizer
    verbose: bool = False

    def estimate(self, **kwargs):
        raise NotImplementedError


#! Addition in the represntation space
@dataclass(frozen=False, kw_only=True)
class BasisEstimator(Estimator):
    layer_name: str

    def __init__(
        self,
        mt: ModelandTokenizer,
        layer_name: str,
        verbose: bool = False,
    ) -> None:
        self.mt = mt
        self.layer_name = layer_name
        self.verbose = verbose

    def estimate(
        self,
        z: str,
        lr: float = 1e-2,
        n_steps: int = 100,
        weight_decay: float = 0,
        optimize_for_tokens: Literal["first", "all"] = "first",
    ) -> torch.Tensor:
        token_indices = self.mt.tokenizer(
            [z], add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

        if optimize_for_tokens == "all":
            raise NotImplementedError

        elif optimize_for_tokens == "first":
            token_idx = token_indices[0]
            if self.verbose:
                logger.info(
                    f"optimizing for token `{self.mt.tokenizer.decode(token_idx)}` [{token_idx}]]"
                )

            lm_head_weights = get_module_nnsight(self.mt, self.mt.lm_head_name).weight
            h_target = lm_head_weights[token_idx].detach()
            # basis = h_final.clone().requires_grad_(True)
            # basis = lm_head_weights.mean(dim=0).detach().requires_grad_(True)
            basis = torch.randn_like(h_target).requires_grad_(True)
            # print(f"BASIS: {project_to_vocab(self.mt, basis, self.layer_name)}")
            # print(
            #     f"TARGET: {project_to_vocab(self.mt, h_target, self.mt.layer_names[-1])}"
            # )

            optimizer = torch.optim.AdamW([basis], lr=lr, weight_decay=weight_decay)
            self.mt._model.train()

            loss_track = []
            progress_bar = tqdm(range(n_steps)) if self.verbose else range(n_steps)

            def patched_run(patch: torch.Tensor, layer_name: str = self.layer_name):
                with self.mt.trace(token_indices) as tr:
                    module = get_module_nnsight(self.mt, layer_name)
                    module.output[0][0, :] = patch
                    lnf = get_module_nnsight(self.mt, self.mt.final_layer_norm_name)
                    lnf_input = lnf.input[0].save()
                    logits = self.mt.output.logits[0, -1].save()
                return lnf_input, logits

            _, logits_target = patched_run(h_target, self.mt.layer_names[-1])
            logits_target = logits_target.detach()

            for iter in progress_bar:
                lnf_input, logits = patched_run(basis)

                # MSE => make the lnf_output as close to the embedding as possible
                # ! limited number of params is making it hard to optimize
                # loss = (lnf_input - h_target).square().sum()

                # loss = (logits - logits_target).square().sum()

                # proba = torch.softmax(logits, dim=-1)
                loss = torch.nn.functional.cross_entropy(
                    logits[None],
                    # logits_target[None],  # match the whole distribution #! doesn't work / takes too long to work
                    torch.Tensor([token_idx])
                    .long()
                    .to(self.mt.device),  # optimize for a single token
                )
                # TODO: generalize to n-way => just adding individual losses should work?

                # MSE on a target logit
                # loss = (logits[token_idx] - 20.0).square().sum()

                loss_track.append(loss.item())

                optimizer.zero_grad()
                self.mt._model.zero_grad()

                loss.backward()
                optimizer.step()

            # print(f"final loss: {loss.item()}")
            # print(f"BASIS: {project_to_vocab(self.mt, basis, self.layer_name)}")

            self.mt._model.eval()
            if self.verbose:
                plt.rcdefaults()
                plt.plot(loss_track)
                plt.xticks(
                    range(0, len(loss_track), len(loss_track) // 10), rotation=45
                )
                plt.xlabel("steps")
                plt.ylabel("loss")
                plt.show()

            # print(basis)

            return Basis(
                direction=basis.detach(),
                z=z,
                token_idx=[token_idx.item()],
                training_args=dict(
                    layer_name=self.layer_name,
                    lr=lr,
                    n_steps=n_steps,
                    weight_decay=weight_decay,
                    optimize_for_tokens=optimize_for_tokens,
                ),
            )

        else:
            raise ValueError(
                f"Unknown {optimize_for_tokens=}, supported values are `first` and `all`"
            )


# ! addition in the logit space
@dataclass(frozen=False, kw_only=True)
class CornerEstimator(Estimator):
    layer_name: str

    def __init__(
        self,
        mt: ModelandTokenizer,
        layer_name: str,
        token_indices: Optional[list[int]] = None,
        verbose: bool = False,
        prompt: Optional[str] = None,
        placeholder: str = "X",
    ):
        self.mt = mt
        self.layer_name = layer_name
        self.token_indices = (
            list(range(mt.tokenizer.vocab_size))
            if token_indices is None
            else token_indices
        )
        self.verbose = verbose

        self.prompt = prompt
        self.placeholder = placeholder
        if self.prompt is not None:
            self.inputs = prepare_input(
                prompts=self.prompt,
                tokenizer=self.mt,
                return_offsets_mapping=True,
            )
            token_rng = find_token_range(
                string=self.prompt,
                substring=self.placeholder,
                occurrence=-1,
                offset_mapping=self.inputs["offset_mapping"][0],
            )
            self.placeholder_pos = token_rng[1]

        else:
            self.prompt = self.mt.tokenizer.bos_token
            self.placeholder = self.mt.tokenizer.bos_token
            self.inputs = self.mt.tokenizer(
                self.mt.tokenizer.bos_token,
                add_special_tokens=False,
                return_tensors="pt",
            )
            self.placeholder_pos = 0

        logger.info(
            f"{self.prompt} |>> {self.placeholder_pos=} | {self.inputs['input_ids'].shape=}"
        )

    def initialize_with_lm_head_rows(self, token_indices: list[int] | None = None):
        token_indices = self.token_indices if token_indices is None else token_indices
        lm_head_weights = get_module_nnsight(self.mt, self.mt.lm_head_name).weight
        embeds = lm_head_weights[token_indices].detach()
        avg_embd = embeds.mean(dim=0).to(self.mt.device)

        if self.verbose:
            init_pred = project_to_vocab(
                mt=self.mt,
                h=avg_embd,
                layer_name=self.layer_name,
                inputs=self.inputs,
                placeholder_pos=self.placeholder_pos,
            )
            logger.info(f"Initialized to {init_pred}")

        avg_embd.requires_grad = True
        return avg_embd, embeds

    @staticmethod
    def find_hypersphere_center(
        points: torch.Tensor,  # shape: (n_dims x n_points)
        lr: float = 1e-2,
        n_steps: int = 100,
    ) -> torch.Tensor:
        # normalize points
        points = points / points.norm(dim=0)

        # find the center of the hypersphere
        center = points.mean(dim=1).requires_grad_(True)

        # print(f"{center.shape=} | {points.shape=}")

        optimizer = torch.optim.Adam([center], lr=lr)
        for _ in range(n_steps):
            center_normalized = center / center.norm()
            dist = center_normalized @ points

            # dist = (points - center_normalized[None].T).norm(dim=0)
            # print(dist.detach())

            # minimize the variance of the cosine similarity
            loss = dist.var()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return center.detach()

    def estimate(
        self,
        class_indices: list[int] | None = None,
        space: Literal["logit", "prob"] = "logit",
        target_val: float | None = None,
        n_steps: int = 200,
        learning_rate: float = 5e-2,
        weight_decay: float = 5e-4,
        concept_subspace_regularizer: float = 0.0,
    ) -> CornerOperator:
        class_indices = self.token_indices if class_indices is None else class_indices
        corner, Q = self.initialize_with_lm_head_rows(class_indices)
        if space == "logit":
            assert target_val is not None
        elif space == "prob":
            if target_val is not None:
                logger.warning("target_val is ignored in 'prob' space")
            target_val = 1.0 / len(class_indices)

            # default to cross entropy loss
            loss_fn = torch.nn.functional.cross_entropy

        device = self.mt.device
        corner = corner.to(device)
        Q = Q.to(device)

        optimizer = torch.optim.AdamW(
            [corner],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        if concept_subspace_regularizer > 0:
            Q = Q.to(self.mt.device).T
            Q = Q / Q.norm(dim=0)

            # the basis vectors aren't orthogonal here, but it's fine
            projection = Q @ (Q.T @ Q).pinverse() @ Q.T
        else:
            del Q

        self.mt._model.train()
        loss_track = []
        progress_bar = tqdm(range(n_steps)) if self.verbose else range(n_steps)
        for iter in progress_bar:
            with self.mt.trace(self.inputs) as tr:
                module = get_module_nnsight(self.mt, self.layer_name)
                current_state = (
                    module.output.save()
                    if (
                        "mlp" in self.layer_name
                        or self.layer_name == self.mt.embedder_name
                    )
                    else module.output[0].save()
                )
                current_state[:, self.placeholder_pos, :] = corner
                logits = self.mt.output.logits[0, -1].save()

            # print(f"{logits.shape=}, {target.shape=}")
            if space == "prob":
                # * pytorch CrossEntropyLoss expects raw logits https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                # probs = torch.softmax(logits, dim=-1)
                loss = 0
                for i, token_idx in enumerate(class_indices):
                    loss += loss_fn(
                        logits[None], torch.Tensor([token_idx]).long().to(device)
                    )

            elif space == "logit":
                # MSE => only care about the target logits, the rest can be ignored
                loss = (logits[class_indices] - target_val).square().sum()
            else:
                raise ValueError(f"Unknown {space=}")

                # # zero out things in corner_last_layer that are not in the subspace Q
                # # ignore if zero_out_subspace_regularizer is 0
                # if concept_subspace_regularizer > 0:
                #     with self.mt.trace(inputs) as tr:
                #         # last layer
                #         last_layer = get_module_nnsight(self.mt, self.mt.layer_names[-1])
                #         last_layer.output[0][0, :] = corner
                #         # model output
                #         out_layer = get_module_nnsight(self.mt, "model")
                #         h_last_layer = out_layer.output[0].save()

                #     h_last_layer = h_last_layer.to(device)
                #     # project out from last representation anything that is not in the subspace
                #     out_of_subspace_contribution = (
                #         torch.eye(self.mt.n_embd).to(device) - projection
                #     ) @ h_last_layer.squeeze()

                #     out_of_subspace_loss = (
                #         concept_subspace_regularizer
                #         * out_of_subspace_contribution.square().sum()
                #     )
                #     # logger.info(f"{loss.item()=}, {out_of_subspace_loss.item()=}")

                # loss += out_of_subspace_loss

            loss_track.append(loss.item())

            optimizer.zero_grad()
            self.mt._model.zero_grad()

            loss.backward()
            optimizer.step()

        self.mt._model.zero_grad()
        self.mt._model.eval()
        free_gpu_cache()

        corner_interp = project_to_vocab(
            mt=self.mt,
            h=corner,
            layer_name=self.layer_name,
            inputs=self.inputs,
            placeholder_pos=self.placeholder_pos,
        )
        if self.verbose:
            plt.rcdefaults()
            plt.plot(loss_track)
            plt.xticks(range(0, len(loss_track), len(loss_track) // 10), rotation=45)
            plt.xlabel("steps")
            plt.ylabel("loss")
            plt.show()
            logger.info(f"Tuned to: {corner_interp}")

        # self.corner = corner.detach()
        # self.token_indices = token_indices
        return CornerOperator(
            mt=self.mt,
            corner=corner.detach(),
            layer=self.layer_name,
            class_indices=class_indices,
            concept_subspace=(projection if concept_subspace_regularizer > 0 else None),
            training_args=dict(
                target_logit=target_val,
                n_steps=n_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            ),
            corner_interpretation=corner_interp,
        )

    # TODO: check if the jacobian idea works.
    # ! This might be doable only if that is the case.
    # def estimate_multi(
    #     self,
    #     class_indices: torch.Tensor,  # shape: (n_classes, n_tokens)
    #     n_steps: int = 200,
    #     learning_rate: float = 5e-2,
    #     weight_decay: float = 5e-4,
    # ) -> CornerOperator:
    #     corner, Q = self.initialize_with_lm_head_rows(class_indices[:, 1].tolist())

    #     optimizer = torch.optim.AdamW(
    #         [corner],
    #         lr=learning_rate,
    #         weight_decay=weight_decay,
    #     )
