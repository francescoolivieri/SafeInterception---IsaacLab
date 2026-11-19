"""Training metrics and skrl callbacks."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from skrl.agents.torch.ppo import PPO
from skrl.utils.runner.torch import Runner


class PerformanceTracker:
    """Compute environment and frame throughput for logging."""

    def __init__(self, num_envs: int, window_sec: float = 2.0, console_interval_sec: float = 30.0) -> None:
        self._num_envs = max(1, num_envs)
        self._window_sec = max(1e-3, window_sec)
        self._console_interval_sec = max(1e-3, console_interval_sec)
        now = time.perf_counter()
        self._last_time = now
        self._last_console_time = now
        self._last_step = 0
        self.env_steps_per_sec = 0.0
        self.frames_per_sec = 0.0

    def update(self, current_step: int) -> Optional[tuple[dict[str, float], bool]]:
        now = time.perf_counter()
        elapsed = now - self._last_time
        if elapsed < self._window_sec:
            return None
        steps = current_step - self._last_step
        if steps <= 0:
            self._last_time = now
            return None

        self.env_steps_per_sec = steps / max(elapsed, 1e-9)
        self.frames_per_sec = self.env_steps_per_sec * self._num_envs
        self._last_time = now
        self._last_step = current_step

        should_print = now - self._last_console_time >= self._console_interval_sec
        if should_print:
            self._last_console_time = now
        return (
            {"perf/env_steps_per_sec": self.env_steps_per_sec, "perf/frames_per_sec": self.frames_per_sec},
            should_print,
        )


class WandbBridge:
    """Use the run opened by skrl; logging is a no-op when W&B is disabled."""

    def __init__(self):
        self._module = None
        self._step_metric_defined = False

    def _ensure_module(self):
        if self._module is None:
            try:
                import wandb
            except ImportError:
                return None
            self._module = wandb
        return self._module

    def log(self, data, step=None):
        module = self._ensure_module()
        if module is None or module.run is None:
            return
        if not self._step_metric_defined:
            module.define_metric("trainer/step")
            module.define_metric("*", step_metric="trainer/step")
            self._step_metric_defined = True
        module.log({**data, "trainer/step": step})

    def finish(self, exit_code=None):
        module = self._ensure_module()
        if module is not None and module.run is not None:
            module.finish(**({"exit_code": exit_code} if exit_code is not None else {}))


class PPORolloutMetrics:
    """Window means plus episode totals that survive rollout boundaries."""

    def __init__(self, base_env):
        self.env = base_env
        self._returns = None
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)

    def _add(self, name, values):
        self._sums[name] += values.sum().item()
        self._counts[name] += values.numel()

    @torch.no_grad()
    def observe(self, *, rewards, actions, terminated, truncated):
        reward = rewards.flatten()
        done = (terminated | truncated).flatten()
        if self._returns is None:
            self._returns = torch.zeros_like(reward)
            self._lengths = torch.zeros_like(reward)
            self._unsafe = torch.zeros_like(done)
            self._collided = torch.zeros_like(done)
            self._min_margin = torch.full_like(reward, float("inf"))
        safety = self.env.get_last_safety_violation().flatten()
        snapshot = self.env.get_last_step_snapshot()
        self._returns += reward
        self._lengths += 1
        self._unsafe |= safety
        self._collided |= snapshot["collision"].flatten()
        self._min_margin = torch.minimum(self._min_margin, snapshot["minimum_safety_margin"].flatten())
        self._add("Reward/step_mean", reward)
        self._add("Policy/action_abs_mean", actions.abs())
        self._add("Policy/action_saturation_percentage", 100.0 * (actions.abs() > 1.0))
        self._add("Safety/step_violation_percentage", 100.0 * safety)
        for name, value in self.env._last_reward_components.items():
            self._add(f"Rewards/{name}", value)
        distance = torch.linalg.vector_norm(snapshot["ref_pos"] - snapshot["pos_local"], dim=-1)
        self._add("Interception/distance_to_target_mean_m", distance)
        if done.any():
            captured = self.env.get_last_episode_status().flatten()[done] == 1
            lengths = self._lengths[done] * self.env.step_dt
            self._add("Episode/return_mean", self._returns[done])
            self._add("Episode/length_seconds_mean", lengths)
            self._add("Episode/capture_percentage", 100.0 * captured)
            self._add("Episode/safe_rollout_percentage", 100.0 * ~self._unsafe[done])
            self._add("Episode/collision_percentage", 100.0 * self._collided[done])
            self._add("Episode/time_to_capture_seconds_mean", lengths[captured])
            margins = self._min_margin[done]
            self._add("Episode/min_safety_margin_mean_m", margins[torch.isfinite(margins)])
            self._returns[done] = self._lengths[done] = 0
            self._unsafe[done] = self._collided[done] = False
            self._min_margin[done] = float("inf")

    def flush(self):
        payload = {name: value / self._counts[name] for name, value in self._sums.items() if self._counts[name]}
        self._sums.clear()
        self._counts.clear()
        return payload


class TrainingMonitor:
    def __init__(self, base_env, bridge, perf, *, global_num_envs=None):
        self.env, self.bridge, self.perf = base_env, bridge, perf
        self.num_envs = global_num_envs or base_env.num_envs
        self.rollout = PPORolloutMetrics(base_env)

    def _log(self, agent, payload, step):
        for key, value in payload.items():
            agent.track_data(key, value)
        self.bridge.log(payload, step=step)

    def record(self, agent, *, timestep, timesteps, rewards, actions, terminated, truncated, **_):
        self.rollout.observe(rewards=rewards, actions=actions, terminated=terminated, truncated=truncated)
        step = timestep + 1
        if step % agent.cfg.rollouts == 0 or step >= timesteps:
            payload = self.rollout.flush()
            payload.update({f"Curriculum/{k}": v for k, v in self.env.get_curriculum_metrics().items()})
            payload.update({"System/num_envs": self.num_envs, "System/total_frames": step * self.num_envs})
            self._log(agent, payload, step)
        self.performance(agent, step)

    def performance(self, agent, step):
        update = self.perf.update(step)
        if update is None:
            return
        metrics, should_print = update
        self._log(agent, {
            "System/env_steps_per_sec": metrics["perf/env_steps_per_sec"],
            "System/frames_per_sec": metrics["perf/frames_per_sec"],
        }, step)
        if should_print:
            print(f"[Perf] step={step:,} env_steps/s={self.perf.env_steps_per_sec:.1f} "
                  f"frames/s={self.perf.frames_per_sec:.1f}")

    @torch.no_grad()
    def updated(self, agent, timestep, elapsed):
        payload = {key: float(np.mean(values)) for key, values in agent.tracking_data.items()
                   if values and key.startswith(("Loss /", "Policy /", "Learning /"))}
        memory = agent.memory
        observations, states, actions, old_log_prob = [
            memory.get_tensor_by_name(name).flatten(0, 1)
            for name in ("observations", "states", "actions", "log_prob")
        ]
        # act() samples even with taken_actions; diagnostics must not consume the training RNG.
        devices = [agent.device.index] if agent.device.type == "cuda" else []
        # The whole rollout in one forward pass overflows a 10 GB card with the graph policy
        # (48 steps x 2048 envs), so go through it in slices. log_prob is per sample, so the
        # result is the same as one big batch.
        log_prob = []
        with torch.random.fork_rng(devices=devices):
            for start in range(0, observations.shape[0], 8192):
                rows = slice(start, start + 8192)
                _, outputs = agent.policy.act({
                    "observations": agent._observation_preprocessor(observations[rows]),
                    "states": agent._state_preprocessor(states[rows]), "taken_actions": actions[rows],
                }, role="policy")
                log_prob.append(outputs["log_prob"])
        log_ratio = torch.cat(log_prob) - old_log_prob
        ratio = log_ratio.exp()
        diagnostics = {
            "Diagnostics / Approximate KL": ((ratio - 1) - log_ratio).mean().item(),
            "Diagnostics / Clip fraction": ((ratio - 1).abs() > agent.cfg.ratio_clip).float().mean().item(),
        }
        for key, value in diagnostics.items():
            agent.track_data(key, value)
        self.bridge.log({**payload, **diagnostics, "Stats / Algorithm update time (ms)": elapsed * 1000},
                        step=timestep + 1)


class MonitoredPPO(PPO):
    """PPO math stays in skrl; these callbacks only report training progress."""
    monitor = None

    def record_transition(self, **transition):
        super().record_transition(**transition)
        if self.monitor is not None:
            self.monitor.record(self, **transition)

    def update(self, *, timestep, timesteps):
        started = time.perf_counter()
        super().update(timestep=timestep, timesteps=timesteps)
        if self.monitor is not None:
            self.monitor.updated(self, timestep, time.perf_counter() - started)


class PPORunner(Runner):
    def _component(self, name):
        return MonitoredPPO if name.lower() == "ppo" else super()._component(name)
