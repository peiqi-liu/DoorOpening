# This module was split out of multi_pcd_dagger.py to keep that file manageable.
# It defines a mixin that is composed into the Dagger class; it is not a standalone
# class and relies on attributes/methods provided by Dagger and the other mixins.
import math

import torch
from rl_games.algos_torch import torch_ext

from DoorOpening.model.transformer import strip_prefix_from_state_dict


def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    adjusted_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            adjusted_state_dict[key] = value
            continue

        parts = key.split(".")
        parts.insert(2, "_orig_mod")
        key_with_orig_mod = ".".join(parts)
        if key_with_orig_mod in model_state_dict:
            adjusted_state_dict[key_with_orig_mod] = value
            continue

        key_no_orig_mod = key.replace("_orig_mod.", "")
        if key_no_orig_mod in model_state_dict:
            adjusted_state_dict[key_no_orig_mod] = value
            continue

        adjusted_state_dict[key] = value
    return adjusted_state_dict


class CheckpointMixin:
    """Teacher/student weight loading and LR-scheduler helpers for Dagger."""

    def _extract_model_state(self, weights):
        if isinstance(weights, dict):
            if "model" in weights:
                return weights["model"], weights
            if "state_dict" in weights:
                return weights["state_dict"], weights
            if "model_state_dict" in weights:
                return weights["model_state_dict"], weights
        return weights, None

    def _load_checkpoint_state(self, ckpt):
        try:
            return torch_ext.load_checkpoint(ckpt)
        except Exception:
            return torch.load(ckpt, map_location="cpu")

    def set_teacher_weights(self, ckpt, model=None, strict=True, allow_adjust=True):
        if model is None:
            model = self.teacher_model
        if model is None:
            raise RuntimeError("Teacher model is not initialized.")
        weights = self._load_checkpoint_state(ckpt)
        state_dict, meta = self._extract_model_state(weights)
        if allow_adjust:
            state_dict = adjust_state_dict_keys(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=strict)
        if meta is not None and "running_mean_std" in meta:
            model.running_mean_std.load_state_dict(meta["running_mean_std"])

    def load_student_weights(self, ckpt):
        weights = torch.load(ckpt, map_location="cpu")
        state_dict, _ = self._extract_model_state(weights)
        state_dict = strip_prefix_from_state_dict(state_dict)
        self.student_model.load_state_dict(state_dict, strict=False)
        if isinstance(weights, dict):
            if "frame" in weights:
                self.frame = int(weights["frame"])
            if "epoch" in weights:
                self.epoch_num = int(weights["epoch"])
            if "student_update_steps" in weights:
                self.student_update_steps = int(weights["student_update_steps"])

            resume_iteration = weights.get("iteration")
            if resume_iteration is None:
                saved_num_envs = int(weights.get("num_envs_at_save", self.num_envs))
                if saved_num_envs <= 0:
                    saved_num_envs = int(self.num_envs)
                resume_iteration = self.frame // max(1, saved_num_envs)
            if resume_iteration is None and "student_update_steps" in weights:
                resume_iteration = int(weights["student_update_steps"])
            self.resume_iteration = int(resume_iteration)
            self._resumed_from_student_ckpt = True

            curriculum_step_count = weights.get("curriculum_step_count")
            if curriculum_step_count is None:
                curriculum_step_count = self.resume_iteration

            # Curriculum/reset scheduling in DooropeningEnv uses common_step_counter.
            if hasattr(self.ov_env, "common_step_counter"):
                self.ov_env.common_step_counter = int(curriculum_step_count)
            if hasattr(self.ov_env, "set_train_info"):
                self.ov_env.set_train_info(int(self.frame))
            elif hasattr(self.ov_env, "_rlgames_env_frames"):
                self.ov_env._rlgames_env_frames = int(self.frame)

            # A training resume must continue BOTH the optimizer (Adam moments) and the
            # LR schedule from the checkpoint; only policy evaluation ignores them.
            if not self.play_policy:
                if "optimizer_state_dict" in weights:
                    try:
                        self.optimizer.load_state_dict(weights["optimizer_state_dict"])
                    except Exception as exc:
                        if self.rank == 0:
                            print(
                                f"Warning: failed to load optimizer state from '{ckpt}'; "
                                f"optimizer will start fresh: {exc}"
                            )
                elif self.rank == 0:
                    print(
                        f"Warning: checkpoint '{ckpt}' has no optimizer state; "
                        "optimizer starts fresh (Adam moments reset)."
                    )
                self._restore_lr_scheduler_state(weights)

        print(f"Loaded student checkpoint: {ckpt}")
        if self.rank == 0 and self._resumed_from_student_ckpt:
            print(
                "Resuming student training state from checkpoint: "
                f"iteration={self.resume_iteration}, curriculum_step_count={int(curriculum_step_count)}, frame={self.frame}, "
                f"student_update_steps={self.student_update_steps}"
            )
        elif self.rank == 0 and self.play_policy:
            print("Loaded student weights for policy evaluation; optimizer and curriculum state were ignored.")

    def _apply_optimizer_runtime_overrides(self):
        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = float(self.weight_decay)

    def _build_lr_scheduler(self):
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._get_lr_schedule_factor,
        )

    def _get_lr_schedule_factor(self, scheduler_step):
        decay_iters = max(1, int(self.lr_decay_iters))
        min_factor = 0.0 if self.lr <= 0.0 else float(self.min_lr / self.lr)
        progress = min(max(int(scheduler_step) + 1, 0), decay_iters) / decay_iters
        if self.lr_schedule == "linear":
            return float(1.0 + (min_factor - 1.0) * progress)
        if self.lr_schedule == "cosine":
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_factor + (1.0 - min_factor) * cosine_factor)
        raise ValueError(f"Unsupported lr_schedule '{self.lr_schedule}'.")

    def _restore_lr_scheduler_state(self, checkpoint):
        if self.lr_scheduler is None:
            return
        scheduler_state = checkpoint.get("lr_scheduler_state_dict") if isinstance(checkpoint, dict) else None
        if scheduler_state is not None:
            try:
                self.lr_scheduler.load_state_dict(scheduler_state)
                restored_lrs = list(getattr(self.lr_scheduler, "_last_lr", []))
                if restored_lrs:
                    for param_group, restored_lr in zip(self.optimizer.param_groups, restored_lrs):
                        param_group["lr"] = float(restored_lr)
                return
            except Exception as exc:
                if self.rank == 0:
                    print(f"Warning: failed to load LR scheduler state from checkpoint: {exc}")

        # Older checkpoints may not include scheduler state. Reconstruct the
        # scheduler position from the number of completed optimizer updates.
        resume_updates = max(0, int(self.student_update_steps))
        self.lr_scheduler.last_epoch = resume_updates - 1
        self.lr_scheduler._step_count = max(1, resume_updates)
        current_lr = float(self.lr) * float(self._get_lr_schedule_factor(self.lr_scheduler.last_epoch))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        self.lr_scheduler._last_lr = [float(param_group["lr"]) for param_group in self.optimizer.param_groups]

    def _get_current_learning_rate(self):
        if getattr(self, "optimizer", None) is None or not self.optimizer.param_groups:
            return float(self.lr)
        return float(self.optimizer.param_groups[0]["lr"])
