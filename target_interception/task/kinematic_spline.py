"""Evaluate cubic target motion on CPU or GPU using IsaacLab's Warp library.

Cached launches share the runtime's tensors without copying. Those tensors must
remain allocated and be updated in place, including during episode resets.
"""
from typing import Any

import torch
import warp as wp


@wp.kernel
def _evaluate_cubic(
    position: wp.array2d(dtype=Any), motion_velocity: wp.array2d(dtype=Any),
    reference_velocity: wp.array2d(dtype=Any), acceleration: wp.array2d(dtype=Any),
    jerk: wp.array2d(dtype=Any), out_position: wp.array2d(dtype=Any),
    out_velocity: wp.array2d(dtype=Any), out_acceleration: wp.array2d(dtype=Any), tau: Any,
):
    env_id, axis = wp.tid()
    a = acceleration[env_id, axis]
    j = jerk[env_id, axis]
    half = type(tau)(0.5)
    six = type(tau)(6.0)
    out_position[env_id, axis] = position[env_id, axis] + tau * (
        motion_velocity[env_id, axis] + tau * (half * a + tau * j / six)
    )
    out_velocity[env_id, axis] = reference_velocity[env_id, axis] + tau * (a + half * tau * j)
    out_acceleration[env_id, axis] = a + tau * j


# Register both precisions before Warp compiles the module on either device.
for scalar_type in (wp.float32, wp.float64):
    wp.overload(_evaluate_cubic, [wp.array2d(dtype=scalar_type)] * 8 + [scalar_type])


def evaluate_substep(env, tau: float) -> None:
    """Reuse launch arguments for each substep; execute on PyTorch's current stream."""
    if not hasattr(env, "_target_spline_launches"):
        wp.init()
        # The motion velocity drops to zero once the final pose is held, the
        # reference velocity does not, so both go in.
        tensors = (
            env._target_policy_step_pos, env._target_kinematic_step_vel,
            env._target_policy_step_vel, env._target_kinematic_step_acc,
            env._target_kinematic_step_jerk, env._target_command_pos,
            env._target_command_vel, env._target_command_acc,
        )
        env._target_spline_views = [wp.from_torch(t) for t in tensors]
        env._target_spline_launches = {}
        env._target_spline_stream = None

    if env._target_command_pos.is_cuda:
        stream = torch.cuda.current_stream(env.device)
        if env._target_spline_stream is None or stream.cuda_stream != env._target_spline_stream.cuda_stream:
            env._target_spline_stream = wp.stream_from_torch(stream)

    if tau not in env._target_spline_launches:
        env._target_spline_launches[tau] = wp.launch(
            _evaluate_cubic, dim=tuple(env._target_command_pos.shape),
            inputs=env._target_spline_views + [env._target_spline_views[0].dtype(tau)],
            device=env._target_spline_views[0].device, stream=env._target_spline_stream,
            record_cmd=True,
        )
    env._target_spline_launches[tau].launch(env._target_spline_stream)
