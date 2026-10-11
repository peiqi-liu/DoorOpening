"""Distillation-only action-latency overrides."""


def set_distillation_action_latency(env_cfg, latency_range=(1, 2)):
    """Override action latency for distillation without changing task defaults."""
    latency_cfg = getattr(env_cfg, "adr_custom_cfg_dict", {}).get("action_latency")
    if not isinstance(latency_cfg, dict):
        raise ValueError("Expected env_cfg.adr_custom_cfg_dict['action_latency'] to be a dict.")
    min_steps, max_steps = (int(latency_range[0]), int(latency_range[1]))
    if min_steps < 0 or max_steps < min_steps:
        raise ValueError("Distillation action latency must satisfy 0 <= min <= max.")
    latency_cfg["latency_steps"] = (min_steps, max_steps)
    print(f"[INFO] Distillation-only action latency range: {min_steps}-{max_steps} policy steps")
