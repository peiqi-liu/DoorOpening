# CEM/MPC framework for DoorOpening

`source/DoorOpening/control/cem_mpc.py` is a state-safe, receding-horizon sampling controller. It
does not replace the RL teacher. It sits above the Isaac Lab environment and proposes a short action
sequence around the current reference trajectory:

```text
camera/policy observation + ref action window
                |
                v
      CEM distribution over H x action_dim
                |
        batched Isaac Lab rollouts
                |
  tracking + contact + panel penetration costs
                |
       elite mean/std update (repeat K times)
                |
       apply first action, shift, repeat
```

The optimizer includes the policy's base, arm, and gripper-intent actions in one vector. The
gripper action is therefore optimized jointly with the wrist pose rather than patched after IK.
The rollout adapter requires explicit snapshot/restore callbacks. This is important: evaluating CEM
candidates must not advance the live Isaac Lab episode.

Minimal integration shape:

```python
from DoorOpening.control.cem_mpc import (
    CEMMPCConfig, CEMMPCPlanner, IsaacLabDoorRolloutAdapter,
)

planner = CEMMPCPlanner(
    action_dim=env.num_policy_actions,
    config=CEMMPCConfig(horizon=16, population=256, elites=32, iterations=4),
    device=env.device,
)
adapter = IsaacLabDoorRolloutAdapter(
    env,
    snapshot_fn=snapshot_door_and_robot_state,
    restore_fn=restore_door_and_robot_state,
    evaluate_fn=evaluate_candidate_chunk,
    chunk_size=32,
)
action, debug = planner.plan(reference_action_window, adapter)
obs, reward, terminated, truncated, info = env.step(action.unsqueeze(0))
```

`evaluate_candidate_chunk` should return a `RolloutBatch` whose cost includes reference tracking,
gripper/handle separation, panel penetration, contact-force limits, action smoothness, and task
progress. The optimizer exposes auxiliary terms in `debug["terms"]` for diagnosing whether a
failure is caused by IK reachability, contact, or poor reference geometry.

This is intentionally a framework layer rather than an unverified change to the active laptop
teacher or vision5 distillation jobs. The next validation step is a small Isaac Lab batch rollout
with articulation snapshots before enabling it for full trajectory regeneration.
