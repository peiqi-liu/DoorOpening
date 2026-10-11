# Real-robot policy/proprioception recording prompt

Implement a synchronized diagnostic recorder for the real-robot door-opening policy. For every policy inference, record one row/frame with a monotonic timestamp and the following data:

- The exact policy action tensor before deployment scaling, plus the final scaled command sent to the robot; include action names/order and units.
- Measured robot joint positions and velocities at inference time, preserving the same joint order and units used by the policy.
- The complete proprioception-history window actually supplied to the policy (all configured fields, such as `q_arm` and `base_vel`), with each history sample's source timestamp/age and field names/order. Record the post-filter/post-delay values seen by the model, not a reconstructed approximation.
- The current handle/grasp target and relevant policy observation metadata when available, including transform/frame identifiers.
- The outgoing arm/base/gripper command, communication timestamps, and any actuator-side saturation or tracking status.

Keep inference-time observations and outgoing commands distinct. Use a monotonic clock, preserve raw values, and log enough metadata to align policy actions with measured state despite transport and actuator latency. Do not change control behavior or add filtering as part of the instrumentation. Include a small validation that confirms history timestamps are ordered and that each logged field matches the tensor passed into that inference.
