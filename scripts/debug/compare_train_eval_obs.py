import argparse
import math
import pathlib
from collections import OrderedDict

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Compare saved train validation and eval step observation batches.")
    parser.add_argument("--train_snapshot", type=str, required=True)
    parser.add_argument("--eval_snapshot", type=str, required=True)
    return parser.parse_args()


def _resolve_snapshot_pt(snapshot_path):
    path = pathlib.Path(snapshot_path).expanduser().resolve()
    if path.suffix == ".pt":
        return path
    return path.with_suffix(".pt")


def _load_snapshot(snapshot_path):
    return torch.load(_resolve_snapshot_pt(snapshot_path), map_location="cpu")


def _flatten_tensor_tree(value, prefix=""):
    items = OrderedDict()
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(_flatten_tensor_tree(child, child_prefix))
        return items
    if isinstance(value, list):
        for idx, child in enumerate(value):
            items.update(_flatten_tensor_tree(child, f"{prefix}[{idx}]"))
        return items
    if isinstance(value, tuple):
        for idx, child in enumerate(value):
            items.update(_flatten_tensor_tree(child, f"{prefix}[{idx}]"))
        return items
    if isinstance(value, torch.Tensor):
        items[prefix] = value
    return items


def _tensor_stats(tensor):
    tensor = tensor.to(dtype=torch.float32)
    return {
        "shape": tuple(tensor.shape),
        "mean": float(tensor.mean().item()) if tensor.numel() > 0 else 0.0,
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
        "min": float(tensor.min().item()) if tensor.numel() > 0 else 0.0,
        "max": float(tensor.max().item()) if tensor.numel() > 0 else 0.0,
    }


def _normalized_mean_std_difference(train_tensor, eval_tensor):
    train_stats = _tensor_stats(train_tensor)
    eval_stats = _tensor_stats(eval_tensor)
    mean_scale = max(abs(train_stats["mean"]), abs(eval_stats["mean"]), 1e-6)
    std_scale = max(abs(train_stats["std"]), abs(eval_stats["std"]), 1e-6)
    mean_diff = abs(train_stats["mean"] - eval_stats["mean"]) / mean_scale
    std_diff = abs(train_stats["std"] - eval_stats["std"]) / std_scale
    return mean_diff + std_diff, train_stats, eval_stats


def _print_top_obs_differences(train_obs, eval_obs):
    train_flat = _flatten_tensor_tree(train_obs)
    eval_flat = _flatten_tensor_tree(eval_obs)
    print("Obs schema:")
    print(f"  train_only={sorted(set(train_flat) - set(eval_flat))[:20]}")
    print(f"  eval_only={sorted(set(eval_flat) - set(train_flat))[:20]}")

    scored = []
    for key in sorted(set(train_flat) & set(eval_flat)):
        train_tensor = train_flat[key]
        eval_tensor = eval_flat[key]
        if tuple(train_tensor.shape) != tuple(eval_tensor.shape):
            scored.append((math.inf, key, {"shape": tuple(train_tensor.shape)}, {"shape": tuple(eval_tensor.shape)}))
            continue
        score, train_stats, eval_stats = _normalized_mean_std_difference(train_tensor, eval_tensor)
        scored.append((score, key, train_stats, eval_stats))
    scored.sort(key=lambda item: item[0], reverse=True)
    print("Top 20 obs keys by normalized mean/std difference:")
    for score, key, train_stats, eval_stats in scored[:20]:
        print(
            f"  {key}: score={score:.6f} "
            f"train(shape={train_stats['shape']}, mean={train_stats.get('mean')}, std={train_stats.get('std')}, min={train_stats.get('min')}, max={train_stats.get('max')}) "
            f"eval(shape={eval_stats['shape']}, mean={eval_stats.get('mean')}, std={eval_stats.get('std')}, min={eval_stats.get('min')}, max={eval_stats.get('max')})"
        )


def _print_named_stats(train_obs, eval_obs):
    train_flat = _flatten_tensor_tree(train_obs)
    eval_flat = _flatten_tensor_tree(eval_obs)
    interesting = [
        "local_pcd_t",
        "aux_handle_pos_temporal",
        "target_err_arm",
        "target_err_hand",
        "q_arm",
        "q_hand",
        "base_vel",
    ]
    print("Focused tensor stats:")
    for key, train_tensor in train_flat.items():
        if not any(name in key for name in interesting):
            continue
        if key not in eval_flat:
            continue
        train_stats = _tensor_stats(train_tensor)
        eval_stats = _tensor_stats(eval_flat[key])
        print(f"  {key}: train={train_stats} eval={eval_stats}")


def _print_action_teacher_stats(train_batch, eval_batch, action_component_dims):
    train_teacher = train_batch["teacher_mus"]
    eval_teacher = eval_batch["teacher_mus"]
    train_action = train_batch["student_action"]
    eval_action = eval_batch["student_action"]
    print("teacher_mus stats:")
    print(f"  train={_tensor_stats(train_teacher)}")
    print(f"  eval={_tensor_stats(eval_teacher)}")
    print("student_action stats:")
    print(f"  train={_tensor_stats(train_action)}")
    print(f"  eval={_tensor_stats(eval_action)}")
    print(f"per-component loss train_raw_mse={train_batch.get('raw_mse')} eval_raw_mse={eval_batch.get('raw_mse')}")

    start = 0
    for name in ("base", "arm", "hand"):
        dim = int(action_component_dims.get(name, 0))
        if dim <= 0:
            continue
        action_slice = slice(start, start + dim)
        train_mse = torch.nn.functional.mse_loss(
            train_action[:, :, action_slice],
            train_teacher.unsqueeze(1)[:, :, action_slice],
            reduction="mean",
        )
        eval_mse = torch.nn.functional.mse_loss(
            eval_action[:, :, action_slice],
            eval_teacher.unsqueeze(1)[:, :, action_slice],
            reduction="mean",
        )
        print(f"  {name}_mse: train={float(train_mse.item())} eval={float(eval_mse.item())}")
        start += dim


def main():
    args = parse_args()
    train_snapshot = _load_snapshot(args.train_snapshot)
    eval_snapshot = _load_snapshot(args.eval_snapshot)
    train_batch = train_snapshot["primary_batch"]
    eval_batch = eval_snapshot["primary_batch"]
    action_component_dims = train_snapshot.get("action_component_dims", {})
    print(f"train_snapshot={args.train_snapshot}")
    print(f"eval_snapshot={args.eval_snapshot}")
    _print_top_obs_differences(train_batch["student_obs"], eval_batch["student_obs"])
    _print_named_stats(train_batch["student_obs"], eval_batch["student_obs"])
    _print_action_teacher_stats(train_batch, eval_batch, action_component_dims)


if __name__ == "__main__":
    main()
