import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Compare train/eval val-action-loss debug snapshots.")
    parser.add_argument("--train_snapshot", type=str, required=True)
    parser.add_argument("--eval_snapshot", type=str, required=True)
    return parser.parse_args()


def _resolve_snapshot_paths(snapshot_path):
    path = Path(snapshot_path).expanduser().resolve()
    if path.suffix == ".json":
        return path, path.with_suffix(".pt")
    if path.suffix == ".pt":
        return path.with_suffix(".json"), path
    return path.with_suffix(".json"), path.with_suffix(".pt")


def _load_snapshot(snapshot_path):
    json_path, pt_path = _resolve_snapshot_paths(snapshot_path)
    with open(json_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    tensor_payload = torch.load(pt_path, map_location="cpu")
    return summary, tensor_payload


def _compare_obs_stats(train_summary, eval_summary):
    train_stats = train_summary.get("student_obs_stats", {})
    eval_stats = eval_summary.get("student_obs_stats", {})
    train_keys = set(train_stats.keys())
    eval_keys = set(eval_stats.keys())
    print("Obs key diff:")
    print(f"  train_only={sorted(train_keys - eval_keys)[:20]}")
    print(f"  eval_only={sorted(eval_keys - train_keys)[:20]}")

    discrepancies = []
    for key in sorted(train_keys & eval_keys):
        train_entry = train_stats[key]
        eval_entry = eval_stats[key]
        if train_entry.get("shape") != eval_entry.get("shape"):
            discrepancies.append((float("inf"), f"{key}: shape {train_entry.get('shape')} != {eval_entry.get('shape')}"))
            continue
        for stat_name in ("mean", "std", "min", "max"):
            train_value = train_entry.get(stat_name)
            eval_value = eval_entry.get(stat_name)
            if train_value is None or eval_value is None:
                continue
            diff = abs(float(train_value) - float(eval_value))
            discrepancies.append((diff, f"{key}.{stat_name}: train={train_value} eval={eval_value} diff={diff}"))
    discrepancies.sort(key=lambda item: item[0], reverse=True)
    print("Largest obs-stat discrepancies:")
    for _, line in discrepancies[:25]:
        print(f"  {line}")


def _compare_selected_tensors(train_summary, train_tensors, eval_summary, eval_tensors):
    train_envs = train_summary.get("selected_envs", [])
    eval_envs = eval_summary.get("selected_envs", [])
    print("Selected env metadata:")
    print(f"  train={train_envs}")
    print(f"  eval={eval_envs}")

    train_slices = train_tensors.get("student_obs_slices", {})
    eval_slices = eval_tensors.get("student_obs_slices", {})
    common_keys = sorted(set(train_slices.keys()) & set(eval_slices.keys()))
    discrepancies = []
    for key in common_keys:
        train_tensor = train_slices[key]
        eval_tensor = eval_slices[key]
        if tuple(train_tensor.shape) != tuple(eval_tensor.shape):
            discrepancies.append((float("inf"), f"{key}: shape {tuple(train_tensor.shape)} != {tuple(eval_tensor.shape)}"))
            continue
        if train_tensor.numel() == 0:
            continue
        diff = (train_tensor.to(dtype=torch.float32) - eval_tensor.to(dtype=torch.float32)).abs()
        discrepancies.append(
            (
                float(diff.max().item()),
                f"{key}: max_abs={float(diff.max().item()):.6g} mean_abs={float(diff.mean().item()):.6g}",
            )
        )
    discrepancies.sort(key=lambda item: item[0], reverse=True)
    print("Largest selected-env tensor discrepancies:")
    for _, line in discrepancies[:25]:
        print(f"  {line}")


def _compare_action_loss(train_summary, eval_summary):
    train_summaries = train_summary.get("action_loss_summaries", {})
    eval_summaries = eval_summary.get("action_loss_summaries", {})
    common_splits = sorted(set(train_summaries.keys()) & set(eval_summaries.keys()))
    print("Action-loss split comparison:")
    for split_name in common_splits:
        train_entry = train_summaries[split_name]
        eval_entry = eval_summaries[split_name]
        print(
            f"  {split_name}: "
            f"train_raw_mse={train_entry.get('raw_mse')} "
            f"eval_raw_mse={eval_entry.get('raw_mse')} "
            f"train_logged={train_entry.get('logged_action_loss')} "
            f"eval_logged={eval_entry.get('logged_action_loss')}"
        )
        for component in ("base_mse", "arm_mse", "hand_mse"):
            if component in train_entry or component in eval_entry:
                print(
                    f"    {component}: "
                    f"train={train_entry.get(component)} eval={eval_entry.get(component)}"
                )


def _compare_top_level_tensor_stats(train_summary, eval_summary):
    for key in ("student_action_stats", "teacher_mus_stats"):
        train_stats = train_summary.get(key, {})
        eval_stats = eval_summary.get(key, {})
        print(f"{key}:")
        for stat_name in ("shape", "mean", "std", "min", "max", "nan_count", "inf_count"):
            print(
                f"  {stat_name}: train={train_stats.get(stat_name)} "
                f"eval={eval_stats.get(stat_name)}"
            )


def main():
    args = parse_args()
    train_summary, train_tensors = _load_snapshot(args.train_snapshot)
    eval_summary, eval_tensors = _load_snapshot(args.eval_snapshot)
    print(f"Train snapshot: {args.train_snapshot}")
    print(f"Eval snapshot: {args.eval_snapshot}")
    _compare_obs_stats(train_summary, eval_summary)
    _compare_selected_tensors(train_summary, train_tensors, eval_summary, eval_tensors)
    _compare_top_level_tensor_stats(train_summary, eval_summary)
    _compare_action_loss(train_summary, eval_summary)


if __name__ == "__main__":
    main()
