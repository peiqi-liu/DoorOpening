import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from DoorOpening.model.transformer import PCDTransformer, strip_prefix_from_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a saved train/eval batch through a checkpointed student model.")
    parser.add_argument("--student_ckpt", type=str, required=True)
    parser.add_argument("--student_cfg", type=str, default=None)
    parser.add_argument("--snapshot", type=str, required=True)
    parser.add_argument("--force_model_mode", choices=["train", "eval"], default="eval")
    return parser.parse_args()


def _resolve_snapshot_pt(snapshot_path):
    path = pathlib.Path(snapshot_path).expanduser().resolve()
    if path.suffix == ".pt":
        return path
    return path.with_suffix(".pt")


def _resolve_cfg_data(snapshot_payload, student_cfg_path):
    cfg_data = snapshot_payload.get("student_cfg_data")
    if isinstance(cfg_data, dict):
        return cfg_data
    if student_cfg_path is None:
        raise ValueError("Snapshot does not embed student_cfg_data; provide --student_cfg.")
    with open(student_cfg_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Student cfg at '{student_cfg_path}' must be a YAML mapping.")
    return loaded


def _extract_model_cfg(student_cfg_data, snapshot_payload):
    model_cfg = snapshot_payload.get("student_model_cfg")
    if isinstance(model_cfg, dict):
        return model_cfg
    ignored_keys = {
        "dagger",
        "local_pcd_range",
        "x_direction_cutoff",
        "door_pcd_num_points",
        "temporal_obs",
        "observation_lag",
        "push_pull_condition_perturb",
    }
    return {
        key: value
        for key, value in student_cfg_data.items()
        if key not in ignored_keys and not str(key).startswith("_")
    }


def _move_to_device(value, device):
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return value


def _component_slices(action_component_dims):
    start = 0
    result = {}
    for name in ("base", "arm", "hand"):
        dim = int(action_component_dims.get(name, 0))
        result[name] = slice(start, start + dim)
        start += dim
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_path = _resolve_snapshot_pt(args.snapshot)
    snapshot_payload = torch.load(snapshot_path, map_location="cpu")
    primary_batch = snapshot_payload.get("primary_batch")
    if not isinstance(primary_batch, dict):
        raise ValueError(f"Snapshot '{snapshot_path}' does not contain primary_batch data.")

    student_cfg_data = _resolve_cfg_data(snapshot_payload, args.student_cfg)
    student_model_cfg = _extract_model_cfg(student_cfg_data, snapshot_payload)
    model = PCDTransformer(**student_model_cfg).to(device)
    weights = torch.load(args.student_ckpt, map_location="cpu")
    state_dict = weights.get("model_state_dict", weights.get("model", weights.get("state_dict", weights)))
    state_dict = strip_prefix_from_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    if args.force_model_mode == "train":
        model.train(True)
    else:
        model.eval()

    student_obs = _move_to_device(primary_batch["student_obs"], device)
    teacher_mus = _move_to_device(primary_batch["teacher_mus"], device)
    saved_student_action = _move_to_device(primary_batch["student_action"], device)

    with torch.no_grad():
        replay_output = model(student_obs)
    replay_action = replay_output["action"]
    replay_mse = F.mse_loss(replay_action, teacher_mus.unsqueeze(1), reduction="mean")
    action_diff = replay_action - saved_student_action
    action_diff_mse = F.mse_loss(replay_action, saved_student_action, reduction="mean")

    action_component_dims = snapshot_payload.get("action_component_dims", {})
    slices = _component_slices(action_component_dims)
    print(f"snapshot={snapshot_path}")
    print(f"student_ckpt={args.student_ckpt}")
    print(f"model.training={model.training}")
    print(f"saved_train_val_mse={primary_batch.get('raw_mse')}")
    print(f"replay_mse={float(replay_mse.detach().cpu().item())}")
    print(f"mse_between_saved_student_action_and_replayed_action={float(action_diff_mse.detach().cpu().item())}")
    print(f"max_abs_action_diff={float(action_diff.abs().max().detach().cpu().item())}")
    for name in ("base", "arm", "hand"):
        action_slice = slices.get(name)
        if action_slice is None:
            continue
        if action_slice.stop <= action_slice.start:
            continue
        component_mse = F.mse_loss(
            replay_action[:, :, action_slice],
            teacher_mus.unsqueeze(1)[:, :, action_slice],
            reduction="mean",
        )
        print(f"{name}_mse={float(component_mse.detach().cpu().item())}")


if __name__ == "__main__":
    main()
