# Vision9 effort-randomized teacher distillation

- Student: fresh initialization; arm tracking error removed from student proprioception.
- Teacher: Vision9 `v9_effort3`, episode 2300, reward 20453.068.
- Teacher checkpoint SHA256: `ba6410ef554029d5985c2fd642ced22282c41321edb50ea78b2d2a649478acd1`.
- References: 1,024 regenerated with `compute_waypoint.py` and the 20-degree traverse yaw offset.
- Trajectory manifest SHA256: `b55b2fd95683cb57fa2161b48e835d5337e73109f2c2f8d2eff5e2a9dcff08cf`.
- Distributed setup: two GPUs, 400 environments per GPU, W&B, videos.
