# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Saif Khan. All rights reserved.

import torch
from mmpose.registry import HOOKS

from .base_evolution_plugin import BaseEvolutionPlugin


def load_state_dict(
    model: str,
    state_dict: dict,
    strict: bool = False,
    logger=None,
    fill_mode: str = "zeros",
):
    """
    Load parameters from a state dictionary into a model, padding with flexible initialization.

    Args:
        model (str): The model to which the state_dict will be loaded.
        state_dict (dict): State dictionary from the previous model.
        strict (bool): If True, ensures that the keys in state_dict and model match exactly.
        logger (Any): Optional logger for logging information about the loading process.
        fill_mode (str): Method for initializing extra parameters if layer size changes.
            Options: "zeros", "mean", "std", "uniform", "kaiming", "xavier".

    Raises:
        RuntimeError: If `strict` is True and unexpected or missing keys are found.
        ValueError: If an invalid `fill_mode` is provided.
    """
    own_state = model.state_dict()
    device = next(model.parameters()).device  # Get the device of the model

    unexpected_keys = set(state_dict.keys()) - set(own_state.keys())
    missing_keys = set(own_state.keys()) - set(state_dict.keys())

    for name, param in state_dict.items():
        if param is None or not isinstance(param, torch.Tensor):
            # Skip non-tensor parameters (buffers, None values)
            continue

        if name in own_state:
            model_param = own_state[name]

            if model_param.shape == param.shape:
                model_param.copy_(param.to(model_param.dtype).to(device))
            else:
                # Handle different fill modes for initialization
                if fill_mode == "zeros":
                    resized_param = torch.zeros(
                        model_param.shape, dtype=model_param.dtype, device=device
                    )
                elif fill_mode == "mean":
                    mean_val = param.mean().item() if param.numel() > 0 else 0
                    resized_param = torch.full(
                        model_param.shape,
                        mean_val,
                        dtype=model_param.dtype,
                        device=device,
                    )
                elif fill_mode == "std":
                    mean, std = param.mean().item(), param.std().item()
                    resized_param = torch.normal(
                        mean=mean,
                        std=std,
                        size=model_param.shape,
                        dtype=model_param.dtype,
                        device=device,
                    )
                elif fill_mode == "uniform":
                    min_val, max_val = param.min().item(), param.max().item()
                    resized_param = torch.empty(
                        model_param.shape, dtype=model_param.dtype, device=device
                    ).uniform_(min_val, max_val)
                elif fill_mode == "kaiming":
                    resized_param = torch.empty(
                        model_param.shape, dtype=model_param.dtype, device=device
                    )
                    torch.nn.init.kaiming_normal_(resized_param)
                elif fill_mode == "xavier":
                    resized_param = torch.empty(
                        model_param.shape, dtype=model_param.dtype, device=device
                    )
                    torch.nn.init.xavier_normal_(resized_param)
                else:
                    raise ValueError(
                        f"Invalid fill_mode '{fill_mode}'. Choose from ['zeros', 'mean', 'std', 'uniform', 'kaiming', 'xavier']."
                    )

                # Copy original values and pad missing dimensions
                slices = tuple(
                    slice(0, min(dim_model, dim_pretrained))
                    for dim_model, dim_pretrained in zip(model_param.shape, param.shape)
                )
                resized_param[slices] = param[slices]
                model_param.copy_(resized_param)

                if logger:
                    logger.info(
                        f"{name}: Resized and copied weights using fill_mode '{fill_mode}' to match shape {model_param.shape}."
                    )

        else:
            if strict:
                raise RuntimeError(f"Error: Unexpected key in state_dict: {name}")
            if logger:
                logger.warning(
                    f"Warning: {name} not found in model and will be ignored."
                )

    if missing_keys:
        error_msg = f"Missing keys in model state_dict: {', '.join(missing_keys)}."
        if strict:
            raise RuntimeError(error_msg)
        if logger:
            logger.warning(error_msg)

    if unexpected_keys:
        error_msg = f"Unexpected keys in state_dict: {', '.join(unexpected_keys)}."
        if strict:
            raise RuntimeError(error_msg)
        if logger:
            logger.warning(error_msg)


@HOOKS.register_module()
class DefaultEvolutionPlugin(BaseEvolutionPlugin):
    """
    A plugin that enables model evolution by taking snapshots and dynamically
    adapting the model architecture after each experience.

    Supports continual learning by snapshotting models, transferring weights, and
    changing architectures across experiences.
    """

    def __init__(
        self,
        model_cfgs: dict = {},
        mode="last",
        fill_mode="std",
        copy_weights=True,
        freeze_backbone=False,
    ):
        """
        Initialize the DefaultEvolutionPlugin.

        Args:
            model_cfgs (dict): Mapping from experience indices to model configurations.
            mode (str): Snapshot retention mode ('last' or 'all').
            fill_mode (str): Strategy for initializing new model weights when switching.
            copy_weights (bool): If True, copies weights from the previous model.
            freeze_backbone (bool): If True, freezes the backbone layers of the new model.
        """
        super().__init__(mode=mode)
        assert isinstance(model_cfgs, dict), "model_cfgs must be a dictionary."

        self.model_cfgs = model_cfgs
        self.fill_mode = fill_mode
        self.copy_weights = copy_weights
        self.freeze_backbone = freeze_backbone

    def switch_model(self, runner, experience_index: int):
        """
        Updates the model architecture after an experience.

        Args:
            runner (Any): The object managing the training process.
            experience_index (int): The index of the current experience.
        """
        next_exp = experience_index + 1
        cfg = self.model_cfgs.get(next_exp, None)

        if cfg is None:
            runner.logger.info(
                f"[DefaultEvolutionPlugin] No change: M_{next_exp} == M_{experience_index}"
            )
            return

        runner.logger.info(f"[DefaultEvolutionPlugin] Switching to M_{next_exp}")

        # Change model architecture
        runner.switch_model(cfg)

        # Get state_dict of the last model
        last_state_dict = runner.memory.get("state_dict", None)
        if isinstance(last_state_dict, list) and len(last_state_dict) > 0:
            runner.logger.info(
                f"[DefaultEvolutionPlugin] Multiple snapshots found. Only the last one will be used."
            )
            last_state_dict = last_state_dict[-1]

        # Load previous weights if available
        if self.copy_weights and last_state_dict is not None:
            runner.logger.info(
                f"[DefaultEvolutionPlugin] Initializing M_{next_exp} with M_{experience_index} weights"
            )
            load_state_dict(
                runner.module,
                last_state_dict,
                strict=False,
                logger=runner.logger,
                fill_mode=self.fill_mode,
            )

            # Freeze backbone layers if needed
            if self.freeze_backbone and hasattr(runner.module, "backbone"):
                for param in runner.module.backbone.parameters():
                    param.requires_grad = False
        elif self.copy_weights:
            runner.logger.warning(
                f"[DefaultEvolutionPlugin] No previous state_dict found for weight transfer!"
            )

    def after_experience(self, runner, experience_index: int):
        """
        Hook to save snapshot and evolve model after an experience.

        Args:
            runner (Any): The object managing the model and training process.
            experience_index (int): The index of the current experience in a continual learning scenario.
        """
        super().after_experience(runner, experience_index)
        self.switch_model(runner, experience_index)
