import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        normalize_state: bool = False,
        normalize_action: bool = False,
        action_std=None,
        action_space: str = "delta",
    ):
        super().__init__()
        self.is_normalize_state = normalize_state
        self.is_normalize_action = normalize_action
        self.action_std = action_std
        self.action_space = action_space
        if self.action_space in {"delta", "relative"} and self.is_normalize_action:
            assert self.action_std is not None, (
                "action_std must be provided if normalize_action is True and "
                "action_space is delta or relative"
            )
            self.action_std = torch.tensor(action_std).float()

    @abstractmethod
    def forward_pass(self, obs, target):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, obs):
        raise NotImplementedError

    def normalize_action(self, actions):
        if self.is_normalize_action:
            actions = actions / self.action_std.to(actions.device)
        return actions

    def decode_actions(self, current_angles, actions):
        if self.is_normalize_action and self.action_space in {"delta", "relative"}:
            actions = actions * self.action_std.to(actions.device)

        if self.action_space == "delta":
            abs_actions = torch.zeros_like(actions)
            abs_actions[:, 0] = current_angles.squeeze(1) + actions[:, 0]
            for i in range(1, actions.shape[1]):
                abs_actions[:, i] = abs_actions[:, i - 1] + actions[:, i]
        elif self.action_space == "relative":
            abs_actions = current_angles + actions
        elif self.action_space == "absolute":
            abs_actions = actions
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")

        return abs_actions
