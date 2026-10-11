"""Fast tests for causal, sub-step camera-source timestamps."""

import torch

from DoorOpening.distillation.camera_capture_timing import (
    interpolate_uniform_history,
    sample_capture_time,
)


def test_half_step_interpolates_measured_state_and_door_pose():
    # Newest sample at t=0, older sample at t=-66.7 ms; camera fired at -33.3 ms.
    history = torch.tensor([[[2.0, 4.0], [0.0, 0.0]]])
    sample = interpolate_uniform_history(history, torch.tensor([1.0 / 30.0]), 1.0 / 15.0)
    torch.testing.assert_close(sample, torch.tensor([[1.0, 2.0]]))

    pose_history = torch.tensor([[[[2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]])
    pose = interpolate_uniform_history(pose_history, torch.tensor([1.0 / 30.0]), 1.0 / 15.0)
    torch.testing.assert_close(pose, torch.tensor([[[1.0, 0.0, 0.0]]]))


def test_quaternion_interpolation_uses_shortest_arc():
    # q and -q encode the same pose; interpolation must not collapse to zero.
    history = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0, 0.0]]]])
    sample = interpolate_uniform_history(history, torch.tensor([1.0 / 30.0]), 1.0 / 15.0, quaternion=True)
    torch.testing.assert_close(sample, torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))


def test_camera_step_variation_is_continuous_causal_and_non_decreasing():
    torch.manual_seed(17)
    starts = torch.zeros(20000)
    first, variation = sample_capture_time(1.0, starts, nominal_age_ms=67, variation_ms=67)
    torch.testing.assert_close(first, 1.0 - 0.067 + variation / 1000.0)
    second, _ = sample_capture_time(
        1.0 + 1.0 / 15.0, starts, first, nominal_age_ms=67, variation_ms=67
    )
    gaps_ms = (second - first) * 1000.0
    assert variation.min() > -67 and variation.max() < 67
    assert torch.all(gaps_ms >= 0)
    assert ((gaps_ms > 25) & (gaps_ms < 40)).any()  # roughly 33 ms is possible
    assert ((gaps_ms > 90) & (gaps_ms < 110)).any()  # roughly 100 ms is possible
    assert torch.unique(gaps_ms.round()).numel() > 100  # not quantized to camera-frame multiples
    assert 125.0 < gaps_ms.quantile(0.9).item() < 155.0
    assert torch.all(second <= 1.0 + 1.0 / 15.0)


def test_capture_time_clamps_to_episode_start():
    start = torch.tensor([10.0])
    capture, _ = sample_capture_time(10.0, start, nominal_age_ms=67, variation_ms=67)
    torch.testing.assert_close(capture, start)
