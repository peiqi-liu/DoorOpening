# Vision5 current-reference distillation

- Teacher: `checkpoints/current_ref_20261111/v9_mesh_best.pth`, SHA-256 `1eb81ed8bd89a39cf9806320210fd39c9caec06e23e274f762b54d81d2c277a9`.
- Teacher source: glorbot2-laptop `v9_mesh` run, commit `59a9a1b6`, trained on the regenerated Vision9 reference trajectories without Franka effort-limit randomization.
- Trajectories: `PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated`, 1,024 `traj.pkl` files copied from Vision9 after the Nov 11 regeneration. Relative-path SHA-256 manifest digest: `b3802e167509640ebfbce7c6907bfd5ddd02e7c740a275548be06743d8866a42`.
- Previous Vision5 copy, dated Nov 7, is preserved under the sibling `_v5_pre_tune_20261111` folder. The stopped `v5_bias_distill` run used that older copy and an earlier Vision9 teacher.
- Teacher audit on Vision5, first 32 doors, one episode each: `v9_mesh` traversed 16/32; `v9_effort3` traversed 11/32. See `analysis/v5_current_ref_teacher_audit_20261111/`.
- Launch: two GPUs, 300 environments per GPU, W&B and videos on both ranks.
