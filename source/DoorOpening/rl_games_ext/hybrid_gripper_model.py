"""Custom rl_games PPO model: Gaussian for every continuous action dim, Bernoulli for the gripper.

Laptop-only experimental trial, NOT committed.

Why this exists: the gripper is commanded as a binary open/closed decision (see
multi_dooropening_env.py's discrete-gripper thresholding, `finger_action > 0.0`), but the stock
rl_games `continuous_a2c_logstd` model treats every action dim -- including the gripper -- as a
continuous Gaussian, sampled and log-prob'd like the rest. That's a real mismatch: PPO's gradient
and entropy for that dimension get computed against a continuous density, not against the actual
discrete outcome being taken, and early in training most samples land near the mu=0 threshold
where the flip is dominated by noise rather than anything learned.

This subclass keeps the base model's `build()` (and therefore the whole a2c_network / observation
normalization / value head machinery) untouched, and only overrides the action distribution: the
network's raw output at the gripper's action index is reinterpreted as a Bernoulli LOGIT
(`p = sigmoid(logit)`) instead of a Gaussian mean, sampled and log-prob'd as a Bernoulli, then
combined with the Normal distribution over the remaining (continuous) dims. The gripper's own
logstd/sigma output is simply unused -- wasted capacity, not incorrect.

The env-side threshold code needs ZERO changes: a Bernoulli sample of 1.0 satisfies
`finger_action > 0.0` (closed) and 0.0 does not (open), exactly matching the existing sign
convention. Action-space bounds in this project are a symmetric [-1, 1] (see `clip_actions: 1.0`
in rl_games_ppo_cfg.yaml), so the clamp+rescale both the player and the training rollout apply to
sampled actions is an identity transform on {0.0, 1.0} -- the discrete sample survives unchanged
through to the env and back through to the stored `prev_actions` used for the PPO update's
neglogp recomputation.
"""

import torch

from rl_games.algos_torch.models import ModelA2CContinuousLogStd

# base(3) + arm(7) + gripper(1) = 11-dim policy action vector; gripper is the last entry.
# See multi_dooropening_env_cfg.py's base_joints/arm_joints/finger_joints lengths and
# multi_dooropening_env.py's _policy_finger_slice, which is built in this exact order.
GRIPPER_ACTION_INDEX = 10


class ModelA2CContinuousLogStdHybridGripper(ModelA2CContinuousLogStd):
    class Network(ModelA2CContinuousLogStd.Network):
        def _split(self, x):
            cont = torch.cat([x[..., :GRIPPER_ACTION_INDEX], x[..., GRIPPER_ACTION_INDEX + 1 :]], dim=-1)
            grip = x[..., GRIPPER_ACTION_INDEX]
            return cont, grip

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)

            mu_cont, grip_logit = self._split(mu)
            sigma_cont, _ = self._split(sigma)
            logstd_cont, _ = self._split(logstd)
            distr_cont = torch.distributions.Normal(mu_cont, sigma_cont, validate_args=False)
            distr_grip = torch.distributions.Bernoulli(logits=grip_logit, validate_args=False)

            if is_train:
                entropy = distr_cont.entropy().sum(dim=-1) + distr_grip.entropy()
                prev_cont, prev_grip = self._split(prev_actions)
                neglogp_cont = self.neglogp(prev_cont, mu_cont, sigma_cont, logstd_cont)
                neglogp_grip = -distr_grip.log_prob(prev_grip)
                result = {
                    "prev_neglogp": torch.squeeze(neglogp_cont + neglogp_grip),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
                return result
            else:
                selected_cont = distr_cont.sample()
                selected_grip = distr_grip.sample()
                selected_action = torch.cat(
                    [
                        selected_cont[..., :GRIPPER_ACTION_INDEX],
                        selected_grip.unsqueeze(-1),
                        selected_cont[..., GRIPPER_ACTION_INDEX:],
                    ],
                    dim=-1,
                )
                neglogp_cont = self.neglogp(selected_cont, mu_cont, sigma_cont, logstd_cont)
                neglogp_grip = -distr_grip.log_prob(selected_grip)
                result = {
                    "neglogpacs": torch.squeeze(neglogp_cont + neglogp_grip),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
                return result


def register():
    from rl_games.algos_torch import model_builder

    model_builder.register_model(
        "continuous_a2c_logstd_hybrid_gripper",
        ModelA2CContinuousLogStdHybridGripper,
    )
