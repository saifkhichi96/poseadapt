# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Saif Khan. All rights reserved.

from fnmatch import fnmatch
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from mmpose.registry import HOOKS
from torch.nn.modules.batchnorm import _NormBase

from poseadapt.engine import BasePlugin
from poseadapt.third_party.avalanche import ParamData, get_layers_and_params

ParamDict = Dict[str, ParamData]
DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class SIPlugin(BasePlugin):
    """
    Synaptic Intelligence (SI), Zenke et al. (2017).
    - Accumulate path integral: trajectory += g ⊙ Δθ  (per mini-batch)
    - At experience end: Ω_i += c * trajectory_i / (Δθ_i^2 + eps)
    - Regularizer during next experience: (λ/2) * Σ_i Ω_i (θ_i - θ_i*)^2

    Args:
        alpha: float or list[float]; λ per experience (last is reused).
        eps: denominator damping.
        excluded_parameters: optional name patterns to exclude (supports wildcards).
        clip_to: clamp Ω to [0, clip_to] to avoid explosion.
        accum_c: multiplier "c" applied to the per-parameter fraction at exp end.
    """

    def __init__(
        self,
        alpha: Union[float, Sequence[float]] = 0.1,
        eps: float = 1e-6,
        excluded_parameters: Optional[Sequence[str]] = None,
        clip_to: float = 1e-3,
        accum_c: float = 1.0,
    ):
        super().__init__()
        self.alpha: List[float] = (
            list(alpha) if isinstance(alpha, (list, tuple)) else [float(alpha)]
        )
        self.eps: float = float(eps)
        self.clip_to: float = float(clip_to)
        self.accum_c: float = float(accum_c)

        self.excluded_parameters: Set[str] = set(excluded_parameters or [])

        # EWC-style storage:
        #   ewc_data[0][name] = θ* (parameter value at last task minimum)
        #   ewc_data[1][name] = Ω  (importance)
        self.ewc_data: Tuple[ParamDict, ParamDict] = (dict(), dict())

        # Per-iteration accumulators for SI
        self.syn_data = {
            "old_theta": dict(),  # θ_t   (flattened)
            "new_theta": dict(),  # θ_{t+1}
            "grad": dict(),  # g_t
            "trajectory": dict(),  # sum_t g_t ⊙ (θ_{t+1} - θ_t)
            "cum_trajectory": dict(),  # accumulated Ω across exps (unused directly here)
        }

    # ---------- hook entry points ----------

    def before_experience(self, runner, experience_index):
        super().before_experience(runner, experience_index)
        model = runner.module
        self._ensure_param_slots(
            model
        )  # allocate ParamData slots & expand on shape changes
        self._init_experience_accumulators(model)

    def before_backward(self, runner, experience_index, losses, data_batch=None):
        super().before_backward(runner, experience_index, losses, data_batch)
        # SI penalty applies from experience 1 onward (Ω exists from previous exps)
        alpha = self._alpha_for_exp(experience_index)
        if alpha <= 0:
            return

        reg = self._si_regularizer(runner.module, device=runner.device, alpha=alpha)
        if reg is not None:
            losses["loss_si"] = reg

    def before_train_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None):
        super().before_train_iter(runner, batch_idx, data_batch)
        # Snapshot θ_t before the train step
        self._snapshot_weights(runner.module, self.syn_data["old_theta"])

    def after_backward(self, runner, batch_idx: int, data_batch: DATA_BATCH = None):
        # Read grads from the step just computed (before optimizer.step)
        self._snapshot_grads(runner.module, self.syn_data["grad"])

    def after_train_iter(
        self, runner, batch_idx: int, data_batch: DATA_BATCH, outputs=None
    ):
        # Snapshot θ_{t+1} after optimizer.step and update path integral
        self._snapshot_weights(runner.module, self.syn_data["new_theta"])
        self._accumulate_trajectory()

    def after_experience(self, runner, experience_index):
        super().after_experience(runner, experience_index)
        # Finalize Ω using current trajectory and Δθ since last optimum θ*
        self._finalize_importance(runner.module)
        # Update θ* to the new optimum
        self._snapshot_weights(runner.module, self.ewc_data[0])

    # ---------- core logic ----------

    def _alpha_for_exp(self, exp_id: int) -> float:
        return self.alpha[exp_id] if exp_id < len(self.alpha) else self.alpha[-1]

    def _ensure_param_slots(self, model):
        # Ensure we have ParamData buffers for each allowed param,
        # and expand them if shapes changed (e.g., head growth).
        for name, p in self._allowed_params(model):
            flat_shape = p.flatten().shape
            if name not in self.ewc_data[0]:
                self.ewc_data[0][name] = ParamData(name, flat_shape)
                self.ewc_data[1][name] = ParamData(f"imp_{name}", flat_shape)
                self.syn_data["old_theta"][name] = ParamData(f"old_{name}", flat_shape)
                self.syn_data["new_theta"][name] = ParamData(f"new_{name}", flat_shape)
                self.syn_data["grad"][name] = ParamData(f"grad_{name}", flat_shape)
                self.syn_data["trajectory"][name] = ParamData(
                    f"traj_{name}", flat_shape
                )
                self.syn_data["cum_trajectory"][name] = ParamData(
                    f"cumtraj_{name}", flat_shape
                )
            else:
                # Expand all slots if the parameter shape has changed
                for store in (
                    self.ewc_data[0],
                    self.ewc_data[1],
                    self.syn_data["old_theta"],
                    self.syn_data["new_theta"],
                    self.syn_data["grad"],
                    self.syn_data["trajectory"],
                    self.syn_data["cum_trajectory"],
                ):
                    store[name].expand(flat_shape)

    @torch.no_grad()
    def _init_experience_accumulators(self, model):
        # θ* (reference) = current weights at start of experience if empty
        self._snapshot_weights(model, self.ewc_data[0])
        # reset trajectory accumulator
        for v in self.syn_data["trajectory"].values():
            v.data.fill_(0.0)

    @torch.no_grad()
    def _snapshot_weights(self, model, target: ParamDict):
        for name, p in self._allowed_params(model):
            if name in target and target[name].shape == p.flatten().shape:
                target[name].data = p.detach().flatten().cpu()

    @torch.no_grad()
    def _snapshot_grads(self, model, target: ParamDict):
        for name, p in self._allowed_params(model):
            g = p.grad
            if g is None:
                # No gradient for this param in the current step
                target[name].data = torch.zeros_like(target[name].data)
            else:
                target[name].data = g.detach().flatten().cpu()

    @torch.no_grad()
    def _accumulate_trajectory(self):
        # trajectory += g_t ⊙ (θ_{t+1} - θ_t)
        for name in self.syn_data["trajectory"]:
            g = self.syn_data["grad"][name].data
            delta = (
                self.syn_data["new_theta"][name].data
                - self.syn_data["old_theta"][name].data
            )
            self.syn_data["trajectory"][name].data.add_(g * delta)

    @torch.no_grad()
    def _finalize_importance(self, model):
        # Update Ω using path integral normalized by squared displacement from θ*
        for name in self.syn_data["cum_trajectory"]:
            traj = self.syn_data["trajectory"][name].data
            theta_new = self.syn_data["new_theta"][name].data
            theta_star = self.ewc_data[0][name].data
            denom = torch.square(theta_new - theta_star) + self.eps
            incr = self.accum_c * (traj / denom)
            # Accumulate in a running container if you want cumulative Ω across exps
            self.syn_data["cum_trajectory"][name].data.add_(incr)

        # Set Ω = clamp_+[0, clip_to](cum_trajectory)
        for name in self.syn_data["cum_trajectory"]:
            omega = self.syn_data["cum_trajectory"][name].data
            omega = torch.clamp(omega, min=0.0, max=self.clip_to)
            self.ewc_data[1][name].data = omega.clone()

    def _si_regularizer(self, model, device, alpha: float):
        loss = None
        for name, p in self._allowed_params(model):
            theta = p.flatten()  # live tensor (no detach)
            theta_star = self.ewc_data[0][name].data.to(device)
            omega = self.ewc_data[1][name].data.to(device)  # Ω ≥ 0
            term = 0.5 * alpha * torch.dot(omega, torch.square(theta - theta_star))
            loss = term if loss is None else loss + term
        return loss

    # ---------- parameter filtering ----------

    def _allowed_params(self, model):
        """Parameters to regularize: requires_grad, not excluded, exclude BN."""
        excluded = self._explode_excluded()
        # exclude BatchNorm params by name (coarse) using Avalanche helper
        for lp in get_layers_and_params(model):
            if isinstance(lp.layer, _NormBase):
                excluded.add(lp.parameter_name)
                excluded.add(lp.parameter_name + ".*")

        out = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(fnmatch(name, pat) for pat in excluded):
                continue
            out.append((name, p))
        return out

    def _explode_excluded(self) -> Set[str]:
        res = set()
        for x in self.excluded_parameters:
            res.add(x)
            if not x.endswith("*"):
                res.add(x + ".*")
        return res
