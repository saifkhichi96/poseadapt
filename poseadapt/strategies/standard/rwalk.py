# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Saif Khan. All rights reserved.

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmpose.registry import HOOKS

from poseadapt.engine import BasePlugin
from poseadapt.third_party.avalanche import (
    ParamData,
    copy_params_dict,
    zerolike_params_dict,
)


@HOOKS.register_module()
class RWalkPlugin(BasePlugin):
    """
    Riemannian Walk (RWalk).
    Two ingredients:
      1) EWC++: EMA of squared gradients to estimate per-parameter importance.
      2) Path-based score s_{t1}^{t2}: first-order loss variation normalized by
         local quadratic form.

    We accumulate both across Δt iterations and across experiences, and then
    regularize future training like EWC with a *combined* penalty.

    Args:
        alpha: weight for the final quadratic penalty.
        decay_factor: EMA factor for importance (higher = more weight on recent).
        delta_t: update the path-based score every `delta_t` iterations.
    """

    def __init__(
        self, alpha: float = 0.1, decay_factor: float = 0.9, delta_t: int = 10
    ):
        super().__init__()
        assert 0.0 <= decay_factor <= 1.0, "`decay_factor` must be in [0, 1]."
        assert delta_t >= 1, "`delta_t` must be at least 1."

        self.decay_factor = float(decay_factor)
        self.alpha = float(alpha)
        self.delta_t = int(delta_t)

        # Checkpointed accumulators for a Δt window
        self.checkpoint_params: Dict[str, ParamData] = dict()  # θ at last checkpoint
        self.checkpoint_loss: Dict[str, ParamData] = dict()  # ΔL accumulators
        self.checkpoint_scores: Dict[
            str, ParamData
        ] = dict()  # s_{t1}^{t2} accumulators

        # Per-iteration storage
        self.iter_params: Dict[str, ParamData] = dict()  # θ_t snapshot
        self.iter_grad: Dict[str, ParamData] = dict()  # grads at step t
        self.iter_importance: Optional[Dict[str, ParamData]] = None  # EMA(|g|^2)

        # Per-experience data
        self.exp_scores: Optional[Dict[str, ParamData]] = None
        self.exp_importance: Optional[Dict[str, ParamData]] = None
        self.exp_params: Optional[Dict[str, ParamData]] = None
        self.exp_penalties: Optional[Dict[str, ParamData]] = None

    def before_train(self, runner):
        # Initialize Δt-window accumulators at the start of the (overall) training run
        self.checkpoint_loss = zerolike_params_dict(runner.module)
        self.checkpoint_scores = zerolike_params_dict(runner.module)
        self.checkpoint_params = copy_params_dict(runner.module)

    def before_train_iter(self, runner, batch_idx, data_batch):
        # Snapshot θ_t (before the optimizer step)
        self.iter_params = copy_params_dict(runner.module)

    def after_backward(self, runner, batch_idx, data_batch=None):
        # Capture gradients from the *current* train step (before optimizer.step)
        self.iter_grad = copy_params_dict(runner.module, copy_grad=True)

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        # 1) Update Taylor ΔL using new θ (after optimizer.step)
        self._update_loss(runner)

        # 2) Update EMA of squared gradients (EWC++)
        self._update_importance()

        # 3) If we reached a Δt boundary, update normalized path scores and reset
        if self._is_checkpoint_iter(runner):
            self._update_score(runner)
            self.checkpoint_loss = zerolike_params_dict(runner.module)
            self.checkpoint_params = copy_params_dict(runner.module)

    def before_backward(self, runner, experience_index, losses, data_batch=None):
        # Apply the quadratic penalty once we have previous experience stats
        if (
            experience_index > 0
            and self.exp_penalties is not None
            and self.exp_params is not None
        ):
            ewc_loss = (
                runner.module.weight.new_tensor(0.0)
                if hasattr(runner.module, "weight")
                else torch.tensor(0.0, device=runner.device)
            )
            for k, p in runner.module.named_parameters():
                if k not in self.exp_penalties:
                    continue  # new params do not count
                pen = self.exp_penalties[k]  # (normalized imp + normalized score)
                p_ref = self.exp_params[k]  # θ* from last exp
                ewc_loss = (
                    ewc_loss
                    + (pen.expand(p.shape) * (p - p_ref.expand(p.shape)).pow(2)).sum()
                )
            losses["loss_rwalk"] = self.alpha * ewc_loss

    def after_experience(self, runner, experience_index: int):
        # Freeze per-experience stats to be used as θ* and penalties in the next exp
        if self.iter_importance is None:
            # No grads (edge case). Create zero tensors matching current params.
            self.iter_importance = zerolike_params_dict(runner.module)

        self.exp_importance = self.iter_importance
        self.exp_params = copy_params_dict(runner.module)

        # Aggregate scores across experiences (moving average of checkpoints)
        if self.exp_scores is None:
            self.exp_scores = self.checkpoint_scores
        else:
            new_scores = {}
            for k, cp in self.checkpoint_scores.items():
                if k not in self.exp_scores:
                    # unseen param earlier; skip or take cp directly
                    new_scores[k] = ParamData(
                        k, device=cp.device, init_tensor=cp.data.clone()
                    )
                else:
                    prev = self.exp_scores[k]
                    shape = cp.data.shape
                    new_scores[k] = ParamData(
                        k,
                        device=prev.device,
                        init_tensor=0.5 * (prev.expand(shape) + cp.data),
                    )
            self.exp_scores = new_scores

        # Build parameter-wise penalties = normalized Ω + normalized ReLU(score)
        self.exp_penalties = {}
        # Avoid divide-by-zero
        max_imp = max(
            (v.data.max() for v in self.exp_importance.values()),
            default=torch.tensor(0.0),
        )
        max_score = max(
            (v.data.max() for v in self.exp_scores.values()), default=torch.tensor(0.0)
        )

        # Put scalars on CPU to avoid device mismatch in comparisons
        max_imp_val = float(max_imp) if torch.is_tensor(max_imp) else max_imp
        max_score_val = float(max_score) if torch.is_tensor(max_score) else max_score

        for k, imp in self.exp_importance.items():
            # Some params may not have scores or vice versa
            score = self.exp_scores.get(k, None)
            imp_term = (
                imp.data / max_imp_val
                if max_imp_val > 0.0
                else torch.zeros_like(imp.data)
            )
            if score is not None:
                scr = F.relu(score.expand(imp.data.shape))
                scr_term = (
                    scr / max_score_val
                    if max_score_val > 0.0
                    else torch.zeros_like(scr)
                )
            else:
                scr_term = torch.zeros_like(imp.data)

            self.exp_penalties[k] = ParamData(
                k, device=imp.device, init_tensor=(imp_term + scr_term)
            )

        # Reset Δt accumulators for the next experience
        self.checkpoint_scores = zerolike_params_dict(runner.module)

    def _is_checkpoint_iter(self, runner) -> bool:
        # In MMEngine, after_train_iter is called before the loop increments _iter.
        # Use (iter + 1) to check boundary at the end of this iteration.
        return ((runner.iter + 1) % self.delta_t) == 0

    @torch.no_grad()
    def _update_loss(self, runner):
        # First-order Taylor approx over one step: ΔL_t ≈ - g_t ⊙ (θ_{t+1} - θ_t)
        for k, p_new in runner.module.named_parameters():
            if k not in self.iter_grad or k not in self.iter_params:
                continue
            p_old = self.iter_params[k]
            g = self.iter_grad[k]
            shape = p_new.shape
            self.checkpoint_loss[k].expand(shape)
            self.checkpoint_loss[k].data.add_(
                -g.expand(shape) * (p_new - p_old.expand(shape))
            )

    def _update_importance(self):
        # EWC++: EMA of squared gradients
        new_imp = {}
        for k, g in self.iter_grad.items():
            new_imp[k] = ParamData(k, device=g.device, init_tensor=g.data.pow(2))

        if self.iter_importance is None:
            self.iter_importance = new_imp
        else:
            updated = {}
            for k, imp_now in new_imp.items():
                if k not in self.iter_importance:
                    updated[k] = ParamData(
                        k, device=imp_now.device, init_tensor=imp_now.data
                    )
                else:
                    old = self.iter_importance[k]
                    updated[k] = ParamData(
                        k,
                        device=imp_now.device,
                        init_tensor=self.decay_factor * imp_now.data
                        + (1 - self.decay_factor) * old.expand(imp_now.shape),
                    )
            self.iter_importance = updated

    @torch.no_grad()
    def _update_score(self, runner):
        # s_{t1}^{t2} += ΔL / (0.5 * Ω * Δθ^2 + eps)
        eps = torch.finfo(torch.float32).eps
        for k, p_new in runner.module.named_parameters():
            if k not in self.iter_importance:
                continue  # skip new or no-grad params
            p_old = self.checkpoint_params[k]
            loss_acc = self.checkpoint_loss[k]
            imp = self.iter_importance[k]
            shape = p_new.shape
            self.checkpoint_scores[k].expand(shape)
            denom = 0.5 * imp.expand(shape) * (p_new - p_old.expand(shape)).pow(2) + eps
            self.checkpoint_scores[k].data.add_(loss_acc.data / denom)
