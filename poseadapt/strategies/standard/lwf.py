# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Saif Khan. All rights reserved.

from typing import Dict

import torch
from mmpose.registry import HOOKS, TRANSFORMS

from poseadapt.engine import BasePlugin

from ..common import convert_predictions, distillation_loss, get_predictions


@HOOKS.register_module()
class LWFPlugin(BasePlugin):
    """Learning without Forgetting (LwF) Plugin.

    LFL uses a distillation loss between logits of the current and previous
    model to mitigate catastrophic forgetting.

    This plugin does not use task identities.

    Args:
        temperature (float): Temperature for distillation loss.
        alpha (float or list[float]): Weight for distillation loss.
            If a list is provided, it should have the same length as the number
            of experiences. If a single value is provided, it will be used for
            all experiences.
        converters (Dict[Dict]): Dictionary with experience index as key and
            'KeypointConverter' configuration as value. This converter will be
            used to convert the model predictions at the corresponding experience
            index to the format of the model at the previous experience index.
            This is only necessary if the model architecture changes between
            experiences, i.e. when the number of keypoints or the keypoint order
            changes. If the model architecture is the same for all experiences,
            this argument can be omitted.

    See Also:
        - Learning without Forgetting
            https://arxiv.org/abs/1606.09282
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        converters: Dict[int, Dict] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

        # Initialize converters
        self.converters = dict()
        if converters is not None:
            for i, c in converters.items():
                self.converters[i] = TRANSFORMS.build(
                    dict(
                        type="KeypointConverter",
                        num_keypoints=c["num_keypoints"],
                        mapping=c["mapping"],
                    )
                )

    def before_experience(self, runner, experience_index: int):
        super().before_experience(runner, experience_index)
        prev_model = runner.last_model
        if experience_index > 0:
            assert prev_model is not None, (
                "The previous model is required for LwF. Please include a "
                "BaseEvolutionPlugin in your config to save the previous model."
            )

            # Set to training mode
            prev_model.train()

    def after_experience(self, runner, experience_index: int):
        super().after_experience(runner, experience_index)
        prev_model = runner.last_model
        if prev_model is not None:
            prev_model.eval()  # Restore to eval mode

    def before_backward(self, runner, experience_index, losses, data_batch=None):
        """
        Add euclidean loss between prev and current features as penalty
        """
        current_model = runner.module
        prev_model = runner.last_model

        if prev_model is None:
            return

        # Get student predictions
        preds_curr = get_predictions(current_model, data_batch)

        # Convert student predictions to teacher format if necessary
        c = self.converters.get(experience_index, None)
        preds_curr = convert_predictions(preds_curr, c)

        # Get teacher predictions
        with torch.no_grad():
            preds_last = get_predictions(prev_model, data_batch)

        # Get current alpha
        alpha = (
            self.alpha[experience_index]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )

        # Compute LwF penalty
        penalty = torch.tensor(0.0, device=runner.device)
        for p_curr, p_last in zip(preds_curr, preds_last):
            penalty += distillation_loss(p_curr, p_last, self.temperature)
        penalty /= len(preds_curr)

        # Add penalty to losses dictionary
        losses["loss_lwf"] = alpha * penalty
        if "loss_kpt" in losses:
            losses["loss_kpt"] = (1 - alpha) * losses["loss_kpt"]
