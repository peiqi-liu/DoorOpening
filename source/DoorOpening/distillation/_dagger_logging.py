# This module was split out of multi_pcd_dagger.py to keep that file manageable.
# It defines a mixin that is composed into the Dagger class; it is not a standalone
# class and relies on attributes/methods provided by Dagger and the other mixins.
import os
import pathlib
from pathlib import Path

import torch
import torch.distributed as dist

try:
    import wandb
except ImportError:
    wandb = None

from DoorOpening.assets.door.multi_door_cfg import DOOR_FAMILY_NAMES


class LoggingMixin:
    """wandb init/logging, episode metrics, and the main _log routine for Dagger."""

    def _init_wandb(self, summaries_dir):
        summaries_path = pathlib.Path(summaries_dir).resolve()
        summaries_path.mkdir(parents=True, exist_ok=True)

        api_key = self.wandb_cfg.get("api_key") or os.getenv("WANDB_API_KEY")
        configured_project = self.wandb_cfg.get("project") or os.getenv("WANDB_PROJECT")
        entity = self.wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
        name = self.wandb_cfg.get("name") or os.getenv("WANDB_NAME")
        mode = self.wandb_cfg.get("mode") or os.getenv("WANDB_MODE")
        if mode is None:
            mode = "online" if (api_key or configured_project or entity or name) else "offline"

        if api_key and mode == "online":
            wandb.login(key=api_key)

        project = configured_project or "dooropening-pcd-dagger"
        notes = self.wandb_cfg.get("notes") or os.getenv("WANDB_NOTES")
        group = self.wandb_cfg.get("group") or os.getenv("WANDB_GROUP")
        job_type = self.wandb_cfg.get("job_type") or os.getenv("WANDB_JOB_TYPE") or "distillation"
        tags = self.wandb_cfg.get("tags") or os.getenv("WANDB_TAGS")
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

        init_kwargs = {
            "project": project,
            "dir": str(summaries_path),
            "config": self.config,
            "job_type": job_type,
        }
        if entity:
            init_kwargs["entity"] = entity
        if name:
            init_kwargs["name"] = name
        if notes:
            init_kwargs["notes"] = notes
        if group:
            init_kwargs["group"] = group
        if mode:
            init_kwargs["mode"] = mode
        if tags:
            init_kwargs["tags"] = tags

        self.wandb_run = wandb.init(**init_kwargs)

    def _wandb_log(self, metrics, step):
        if not self.use_wandb or self.wandb_run is None or not metrics:
            return
        wandb.log(metrics, step=step)

    def _to_loggable_scalar(self, value):
        if isinstance(value, (bool, int, float)):
            return float(value)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.detach().cpu().item())
        return None

    def _update_logged_env_metrics(self, extras):
        if not isinstance(extras, dict):
            return

        metrics = {}
        for key, value in extras.items():
            if not any(key.startswith(prefix) for prefix in self.logged_env_metric_prefixes):
                continue
            scalar_value = self._to_loggable_scalar(value)
            if scalar_value is None:
                continue
            metrics[key] = scalar_value

        if metrics:
            self.latest_env_log_metrics.update(metrics)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        wandb.finish()
        self.wandb_run = None

    def _mean_completed_metric(self, values):
        if not values:
            return None
        return float(sum(values) / len(values))

    def _update_completed_episode_metrics(self, done_mask, timed_out):
        if done_mask.numel() == 0:
            return

        episode_rewards = self.current_rewards[done_mask, 0].detach().cpu().tolist()
        episode_lengths = self.current_lengths[done_mask].detach().cpu().tolist()
        # Student success = the env's TASK success (reached the last reference frame), matching the
        # teacher's success/success_rate. Read the persisted per-env buffer because env.step() has
        # already cleared the live latch. Fall back to timed_out (survived-to-timeout) if unavailable.
        last_success = getattr(getattr(self, "ov_env", None), "last_success", None)
        if last_success is not None:
            episode_success_tensor = last_success[done_mask].to(dtype=torch.float32)
        else:
            episode_success_tensor = timed_out[done_mask].to(dtype=torch.float32)

        self.completed_rewards.extend(float(value) for value in episode_rewards)
        self.completed_lengths.extend(float(value) for value in episode_lengths)
        self.interval_success_count[done_mask] += episode_success_tensor.to(device=self.device)
        self.interval_completed_count[done_mask] += 1.0

    def _clear_interval_success_rates(self):
        self.interval_success_count.zero_()
        self.interval_completed_count.zero_()

    def _get_global_success_rate_for_mask(self, split_mask, global_target):
        split_mask = split_mask.to(device=self.device, dtype=torch.bool)
        valid_mask = (self.interval_completed_count > 0) & split_mask
        per_env_success_rate = torch.zeros_like(self.interval_success_count)
        per_env_success_rate[valid_mask] = (
            self.interval_success_count[valid_mask] / self.interval_completed_count[valid_mask]
        )

        stats = torch.zeros(2, dtype=torch.float64, device=self.device)
        stats[0] = float(per_env_success_rate[valid_mask].sum().detach().cpu().item())
        stats[1] = float(valid_mask.sum().detach().cpu().item())
        if self.use_ddp:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[1] <= 0:
            return None, False

        is_ready = bool(stats[1].detach().cpu().item() >= global_target)
        if not is_ready:
            return None, False
        return float((stats[0] / stats[1]).detach().cpu().item()), True

    def _get_global_success_rates(self):
        num_families = len(DOOR_FAMILY_NAMES)
        stats = torch.zeros((1 + num_families, 2), dtype=torch.float64, device=self.device)

        valid_mask = (self.interval_completed_count > 0) & self.train_env_mask
        per_env_success_rate = torch.zeros_like(self.interval_success_count)
        per_env_success_rate[valid_mask] = (
            self.interval_success_count[valid_mask] / self.interval_completed_count[valid_mask]
        )
        stats[0, 0] = float(per_env_success_rate[valid_mask].sum().detach().cpu().item())
        stats[0, 1] = float(valid_mask.sum().detach().cpu().item())
        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            family_mask = valid_mask & (self.env_family_ids == int(family_id))
            stats[family_id + 1, 0] = float(per_env_success_rate[family_mask].sum().detach().cpu().item())
            stats[family_id + 1, 1] = float(family_mask.sum().detach().cpu().item())

        if self.use_ddp:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        is_ready = bool(stats[0, 1].detach().cpu().item() >= self.global_train_num_envs)
        success_rate = None
        if is_ready and stats[0, 1] > 0:
            success_rate = float((stats[0, 0] / stats[0, 1]).detach().cpu().item())

        family_success_rates = {}
        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            count = stats[family_id + 1, 1]
            if is_ready and count > 0:
                family_success_rates[family_name] = float(
                    (stats[family_id + 1, 0] / count).detach().cpu().item()
                )
            else:
                family_success_rates[family_name] = None
        validation_success_rate, validation_ready = self._get_global_success_rate_for_mask(
            self.validation_env_mask,
            self.global_validation_num_envs,
        )
        return success_rate, family_success_rates, validation_success_rate, is_ready, validation_ready

    def _log(
        self,
        iteration,
        train_total_loss,
        train_action_loss,
        train_aux_loss,
        train_mode_loss,
        train_door_joint_loss,
        validation_total_loss,
        validation_action_loss,
        teacher_forcing_beta,
    ):
        if iteration % self.log_interval != 0:
            return
        episode_reward = self._mean_completed_metric(self.completed_rewards)
        episode_length = self._mean_completed_metric(self.completed_lengths)
        env_step_dt = max(float(getattr(self.ov_env, "dt", 0.0)), 1e-6)
        episode_length_seconds = episode_length * env_step_dt if episode_length is not None else None
        success_rate = None
        family_success_rates = {}
        validation_success_rate = None
        success_rate_ready = False
        validation_success_rate_ready = False
        if iteration > 0:
            (
                success_rate,
                family_success_rates,
                validation_success_rate,
                success_rate_ready,
                validation_success_rate_ready,
            ) = self._get_global_success_rates()
        teacher_env_fraction = self._get_teacher_forcing_env_fraction()
        student_env_fraction = 1.0 - teacher_env_fraction
        iteration_time_ms = self._consume_timing_means()

        if self.rank == 0:
            print("=" * 10)
            print("ITERATION:", iteration)
            print("Train Total Loss:", float(train_total_loss.detach().cpu()))
            print("Train Action Loss:", float(train_action_loss.detach().cpu()))
            if train_aux_loss is not None:
                print("Train Aux Loss:", float(train_aux_loss.detach().cpu()))
            if train_mode_loss is not None:
                print("Train Direction Loss:", float(train_mode_loss.detach().cpu()))
            if train_door_joint_loss is not None:
                print("Train Door Joint Loss:", float(train_door_joint_loss.detach().cpu()))
                if self.latest_door_joint_abs_err is not None:
                    print("Door Joint Abs Err (rad):", self.latest_door_joint_abs_err)
            if validation_total_loss is not None:
                print("Validation Total Loss:", float(validation_total_loss.detach().cpu()))
            if validation_action_loss is not None:
                print("Validation Action Loss:", float(validation_action_loss.detach().cpu()))
            if self.latest_mode_direction_acc is not None:
                print("Direction Acc:", self.latest_mode_direction_acc)
            if self.mode_prediction_loss_enabled:
                print("Direction Window Acc:", self.latest_dir_window_acc)
                print("Direction Window Balanced Acc:", self.latest_dir_window_balanced_acc)
                print("Direction Window Push Acc:", self.latest_dir_window_push_acc)
                print("Direction Window Pull Acc:", self.latest_dir_window_pull_acc)
                print("Direction Window Num Push Labels:", self.latest_dir_window_num_push_labels)
                print("Direction Window Num Pull Labels:", self.latest_dir_window_num_pull_labels)
                print("Direction Window Num Push Preds:", self.latest_dir_window_num_push_preds)
                print("Direction Window Num Pull Preds:", self.latest_dir_window_num_pull_preds)
            if self.door_joint_prediction_enabled:
                print("Door Joint Target Mean (rad):", self.latest_door_joint_target_mean)
            if self.observation_lag_enabled:
                print("Obs Lag Enabled:", bool(self.latest_obs_lag_enabled))
                print("Obs Lag Mean (ms):", self.latest_obs_lag_mean_ms)
                print("Obs Lag Min (ms):", self.latest_obs_lag_min_ms)
                print("Obs Lag Max (ms):", self.latest_obs_lag_max_ms)
            if self.push_pull_condition_enabled:
                print("Push/Pull Condition Source:", self.latest_push_pull_condition_source)
                print("Fraction Push:", self.latest_fraction_push)
                print("Fraction Pull:", self.latest_fraction_pull)
                print("Push/Pull Pred Entropy:", self.latest_push_pull_pred_entropy)
                if self.latest_push_pull_pred_acc is not None:
                    print("Push/Pull Pred Acc:", self.latest_push_pull_pred_acc)
                print("Push/Pull Perturbed To Push Count:", self.latest_push_pull_perturb_to_push_count)
                print("Push/Pull Perturbed To Pull Count:", self.latest_push_pull_perturb_to_pull_count)
            if self.left_right_condition_enabled:
                print("Fraction Left:", self.latest_fraction_left)
                print("Fraction Right:", self.latest_fraction_right)
            if self.door_hole_aug_enabled and self.latest_door_hole_aug_stats:
                _hole_stats = self.latest_door_hole_aug_stats
                _n_envs = _hole_stats.get("door_hole_aug/num_envs")
                _n_hole = _hole_stats.get("door_hole_aug/hole_env_count")
                _n_reflect = _hole_stats.get("door_hole_aug/reflection_env_count")
                _n_pure = _hole_stats.get("door_hole_aug/pure_hole_env_count")
                print(f"Door Hole Envs (hole / total): {_n_hole} / {_n_envs}")
                if _n_reflect is not None:
                    print(f"Door Mirror Envs (reflective-glass): {_n_reflect}")
                    print(f"Door Bright-Hole Envs (pure hole, no mirror): {_n_pure}")
            print("Temporal Aux Handle Enabled:", bool(self.temporal_aux_handle_enabled))
            print("Temporal Push/Pull Belief Enabled:", bool(self.temporal_push_pull_belief_enabled))
            if self.temporal_push_pull_belief_enabled:
                print("Push/Pull Belief Hist Entropy Now:", self.latest_push_pull_belief_hist_entropy_now)
                print("Push/Pull Belief Hist Entropy Mean:", self.latest_push_pull_belief_hist_entropy_mean)
            print("Teacher Forcing Beta:", teacher_forcing_beta)
            print("Teacher Rollout Env Fraction:", teacher_env_fraction)
            print("Student Rollout Env Fraction:", student_env_fraction)
            print("Student Update Steps:", self.student_update_steps)
            print("Learning Rate:", self._get_current_learning_rate())
            print("Last Local Update Batch Size:", self.last_local_update_batch_size)
            print("Last Global Update Batch Size:", self.last_global_update_batch_size)
            if episode_reward is not None:
                print("Episode Reward:", episode_reward)
            if episode_length is not None:
                print("Episode Length:", episode_length)
            if episode_length_seconds is not None:
                print("Episode Length (s):", episode_length_seconds)
            if success_rate is not None:
                print("Train Success Rate:", success_rate)
            if validation_success_rate is not None:
                print("Validation Success Rate:", validation_success_rate)
            for family_name, family_success_rate in family_success_rates.items():
                if family_success_rate is not None:
                    print(f"Train Success Rate/{family_name}:", family_success_rate)
            if iteration_time_ms is not None:
                print("Iteration Time (ms):", iteration_time_ms)
            # for key, value in sorted(self.latest_env_log_metrics.items()):
            #     print(f"{key}:", value)

        metrics = {
            "loss/total": float(train_total_loss.detach().cpu()),
            "loss/action": float(train_action_loss.detach().cpu()),
            "dist/world_size": self.world_size,
            "dist/update_steps": self.student_update_steps,
            "dist/last_local_update_batch_size": self.last_local_update_batch_size,
            "dist/last_global_update_batch_size": self.last_global_update_batch_size,
            "schedule/learning_rate": self._get_current_learning_rate(),
        }
        if train_aux_loss is not None:
            metrics["loss/aux"] = float(train_aux_loss.detach().cpu())
        if train_mode_loss is not None:
            metrics["loss/direction"] = float(train_mode_loss.detach().cpu())
        if train_door_joint_loss is not None:
            metrics["loss/door_joint"] = float(train_door_joint_loss.detach().cpu())
            if self.latest_door_joint_abs_err is not None:
                for name, err in zip(self.door_joint_prediction_joint_names, self.latest_door_joint_abs_err):
                    metrics[f"stats/door_joint_abs_err_{name}"] = err
        if validation_total_loss is not None:
            metrics["loss/val_total"] = float(validation_total_loss.detach().cpu())
        if validation_action_loss is not None:
            metrics["loss/val_action"] = float(validation_action_loss.detach().cpu())
        if self.latest_mode_direction_acc is not None:
            metrics["stats/dir_acc"] = self.latest_mode_direction_acc
        if self.mode_prediction_loss_enabled:
            metrics["stats/dir_window_acc"] = self.latest_dir_window_acc
            metrics["stats/dir_window_balanced_acc"] = self.latest_dir_window_balanced_acc
            metrics["stats/dir_window_push_acc"] = self.latest_dir_window_push_acc
            metrics["stats/dir_window_pull_acc"] = self.latest_dir_window_pull_acc
            metrics["stats/dir_window_num_push_labels"] = self.latest_dir_window_num_push_labels
            metrics["stats/dir_window_num_pull_labels"] = self.latest_dir_window_num_pull_labels
            metrics["stats/dir_window_num_push_preds"] = self.latest_dir_window_num_push_preds
            metrics["stats/dir_window_num_pull_preds"] = self.latest_dir_window_num_pull_preds
        if self.door_joint_prediction_enabled:
            if self.latest_door_joint_target_mean is not None:
                for name, val in zip(self.door_joint_prediction_joint_names, self.latest_door_joint_target_mean):
                    metrics[f"stats/door_joint_target_mean_{name}"] = val
        if self.observation_lag_enabled:
            metrics["timestamp/obs_lag_enabled"] = self.latest_obs_lag_enabled
            metrics["timestamp/obs_lag_mean_ms"] = self.latest_obs_lag_mean_ms
            metrics["timestamp/obs_lag_min_ms"] = self.latest_obs_lag_min_ms
            metrics["timestamp/obs_lag_max_ms"] = self.latest_obs_lag_max_ms
            for timestamp_ms, mean_age_ms in self.latest_obs_lag_effective_age_ms_by_timestamp.items():
                metrics[f"timestamp/obs_lag_effective_age_{timestamp_ms}ms"] = mean_age_ms
        metrics["stats/temporal_aux_handle_enabled"] = float(self.temporal_aux_handle_enabled)
        metrics["stats/temporal_push_pull_belief_enabled"] = float(self.temporal_push_pull_belief_enabled)
        metrics["stats/push_pull_belief_hist_entropy_now"] = self.latest_push_pull_belief_hist_entropy_now
        metrics["stats/push_pull_belief_hist_entropy_mean"] = self.latest_push_pull_belief_hist_entropy_mean
        if self.push_pull_condition_enabled:
            metrics["stats/push_pull_condition_source"] = self.latest_push_pull_condition_source
            metrics["stats/fraction_push"] = self.latest_fraction_push
            metrics["stats/fraction_pull"] = self.latest_fraction_pull
            metrics["stats/push_pull_pred_entropy"] = self.latest_push_pull_pred_entropy
            if self.latest_push_pull_pred_acc is not None:
                metrics["stats/push_pull_pred_acc"] = self.latest_push_pull_pred_acc
            metrics["stats/push_pull_perturb_to_push_count"] = self.latest_push_pull_perturb_to_push_count
            metrics["stats/push_pull_perturb_to_pull_count"] = self.latest_push_pull_perturb_to_pull_count
        if self.left_right_condition_enabled:
            metrics["stats/fraction_left"] = self.latest_fraction_left
            metrics["stats/fraction_right"] = self.latest_fraction_right
        if episode_reward is not None:
            metrics["stats/episode_reward"] = episode_reward
        if episode_length is not None:
            metrics["stats/episode_length"] = episode_length
        if episode_length_seconds is not None:
            metrics["stats/episode_length_seconds"] = episode_length_seconds
        if success_rate is not None:
            metrics["success/success_rate"] = success_rate
        if validation_success_rate is not None:
            metrics["success/validation_success_rate"] = validation_success_rate
        for family_name, family_success_rate in family_success_rates.items():
            if family_success_rate is not None:
                metrics[f"success/success_rate/{family_name}"] = family_success_rate
        if teacher_forcing_beta is not None:
            metrics["schedule/teacher_forcing_beta"] = teacher_forcing_beta
        metrics["schedule/teacher_rollout_env_fraction"] = teacher_env_fraction
        metrics["schedule/student_rollout_env_fraction"] = student_env_fraction
        if iteration_time_ms is not None:
            metrics["timing/iteration_ms"] = iteration_time_ms
        if self.latest_door_hole_aug_stats:
            for key, value in self.latest_door_hole_aug_stats.items():
                metrics[key] = value
        if getattr(self, "latest_door_frame_aug_stats", None):
            for key, value in self.latest_door_frame_aug_stats.items():
                metrics[key] = value
        if self.latest_env_log_metrics:
            for key, value in self.latest_env_log_metrics.items():
                if key == "stats/success_rate" or key.startswith("stats/success_rate/"):
                    continue
                metrics[key] = value
        self._wandb_log(metrics, step=iteration)
        should_clear_success_rates = success_rate_ready and (
            self.global_validation_num_envs <= 0 or validation_success_rate_ready
        )
        if should_clear_success_rates:
            self._clear_interval_success_rates()
