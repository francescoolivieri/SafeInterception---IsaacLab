"""Configuration of the perturbed base environment (timing, action interface, measurement model)."""

from __future__ import annotations

import math

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class MeasurementModelCfg:
    """Thesis, Table "Measurement uncertainty" (profile atomic_20260811)."""

    sensor_rate_hz: int = 100
    # Phase of the 100 Hz sensor clock on the 500 Hz physics grid, drawn per episode.
    randomize_sensor_phase: bool = True

    position_white_std_m: tuple[float, float, float] = (1.0e-5, 1.0e-5, 1.0e-5)

    velocity_delay_min_s: float = 0.009
    velocity_delay_max_s: float = 0.016
    velocity_episode_bias_std_m_s: tuple[float, float, float] = (0.020, 0.020, 0.005)
    velocity_episode_bias_clip_m_s: tuple[float, float, float] = (0.050, 0.050, 0.015)
    velocity_residual_std_m_s: tuple[float, float, float] = (0.016, 0.016, 0.0065)
    velocity_residual_time_constant_s: tuple[float, float, float] = (0.10, 0.10, 0.10)

    gyro_white_std_rad_s: tuple[float, float, float] = (0.0023136162, 0.0031482635, 0.0022917131)
    gyro_episode_bias_std_rad_s: tuple[float, float, float] = (0.0015, 0.0015, 0.0015)


@configclass
class PerturbedBaseEnvCfg(DirectRLEnvCfg):
    """Thesis, Table "Simulation and control timing parameters"."""

    episode_length_s = 10.0
    sim_frequency = 500
    policy_rate_hz = 50
    decimation = sim_frequency // policy_rate_hz

    sim: SimulationCfg = SimulationCfg(dt=1 / sim_frequency, render_interval=decimation)
    terrain: TerrainImporterCfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # Action: normalised world-frame command [vx, vy, vz, yaw_rate] in [-1, 1], scaled by these.
    action_space = 4
    velocity_command_scale_m_s: tuple[float, float, float] = (2.0, 2.0, 1.0)
    velocity_yaw_rate_scale_rad_s: float = math.pi / 2.0

    # Nominal command delay: the previous command stays applied for this long into each policy step.
    command_delay_s: float = 0.010

    measurement_model: MeasurementModelCfg = MeasurementModelCfg()

    # Vehicle articulation (not included in this project; see PerturbedBaseEnv).
    robot = None
