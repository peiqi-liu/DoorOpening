"""Shared camera-source timing and interpolation for simulated observations.

This is an effective selected-cloud timestamp model, not a synthetic 30 Hz
camera callback. It produces continuous early/late variation around a nominal
source age while preserving causal, nondecreasing selected timestamps.
"""

import torch


def sample_capture_time(
    current_time_s, episode_start_s, previous_capture_time_s=None, *, nominal_age_ms, variation_ms
):
    """Choose a continuously varied source time shared by cloud and sensor state."""
    if nominal_age_ms < 0 or variation_ms < 0 or variation_ms > nominal_age_ms:
        raise ValueError("Invalid nominal capture age or signed variation range.")
    signed_variation_ms = (torch.rand_like(episode_start_s) * 2.0 - 1.0) * float(variation_ms)
    # Positive offset means a later capture. The nominal source is one policy
    # interval old, so even the latest allowed source is available by inference.
    capture_time_s = torch.full_like(episode_start_s, float(current_time_s)) - (
        float(nominal_age_ms) - signed_variation_ms
    ) / 1000.0
    capture_time_s = torch.maximum(capture_time_s, episode_start_s)
    if previous_capture_time_s is not None:
        capture_time_s = torch.maximum(
            capture_time_s,
            torch.where(torch.isfinite(previous_capture_time_s), previous_capture_time_s, episode_start_s),
        )
    return capture_time_s, signed_variation_ms


def interpolate_uniform_history(history, age_s, dt_s, *, quaternion=False):
    """Interpolate newest-first policy-step history at a continuous per-env age.

    Pose tensors may have an extra body dimension. Quaternions use shortest-arc
    normalized linear interpolation in Isaac's wxyz convention.
    """
    if history.ndim < 3 or age_s.ndim != 1 or history.shape[0] != age_s.shape[0] or dt_s <= 0:
        raise ValueError("Expected history [env, time, ...], age [env], and positive dt.")
    phase = (age_s / float(dt_s)).clamp(0.0, history.shape[1] - 1)
    newer_idx = phase.floor().long()
    older_idx = (newer_idx + 1).clamp(max=history.shape[1] - 1)
    weight = (phase - newer_idx).view(-1, *([1] * (history.ndim - 2)))
    env_idx = torch.arange(history.shape[0], device=history.device)
    newer = history[env_idx, newer_idx]
    older = history[env_idx, older_idx]
    if quaternion:
        if history.shape[-1] != 4:
            raise ValueError("Quaternion history must end in four components.")
        older = torch.where((newer * older).sum(dim=-1, keepdim=True) < 0.0, -older, older)
        return torch.nn.functional.normalize((1.0 - weight) * newer + weight * older, dim=-1)
    return (1.0 - weight) * newer + weight * older
