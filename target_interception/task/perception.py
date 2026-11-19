"""Target and obstacle measurements on the same sensor clock as the pursuer."""

from __future__ import annotations

import copy

import torch

from target_interception.sim.measurement_model import MeasurementModel


class AlignedEntityObservationModel:
    """Independent entity noise, with timing copied from the pursuer."""

    def __init__(
        self,
        *,
        num_envs: int,
        num_entities: int,
        device: torch.device | str,
        sim_dt: float,
        policy_dt: float,
        cfg,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_entities = int(num_entities)
        self.device = torch.device(device)
        if self.num_entities < 1:
            raise ValueError("AlignedEntityObservationModel requires at least one entity.")

        # Phase is supplied by the interceptor model. Disable the wrapped
        # model's own draw so every channel remains on one measurement clock.
        entity_cfg = copy.deepcopy(cfg)
        entity_cfg.randomize_sensor_phase = False
        self._model = MeasurementModel(
            num_envs=self.num_envs * self.num_entities,
            device=self.device,
            sim_dt=sim_dt,
            policy_dt=policy_dt,
            cfg=entity_cfg,
        )
        self._entity_offsets = torch.arange(self.num_entities, device=self.device)
        rows = self.num_envs * self.num_entities
        self._identity_quaternion = torch.zeros(rows, 4, device=self.device)
        self._identity_quaternion[:, 0] = 1.0
        self._zero_body_rate = torch.zeros(rows, 3, device=self.device)

    def _flat_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        return (env_ids[:, None] * self.num_entities + self._entity_offsets).reshape(-1)

    def _flatten_entities(self, values: torch.Tensor, rows: int) -> torch.Tensor:
        expected = (rows, self.num_entities, 3)
        if tuple(values.shape) != expected:
            raise ValueError(f"Expected entity tensor with shape {expected}, got {tuple(values.shape)}.")
        return values.reshape(rows * self.num_entities, 3)

    def sync_timing(self, interceptor_model, env_ids: torch.Tensor | None = None) -> None:
        """Copy the interceptor's per-episode phase and channel delays."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        flat_ids = self._flat_ids(env_ids)
        for name in ("sensor_age_steps", "velocity_delay_steps"):
            source = getattr(interceptor_model, name)[env_ids]
            getattr(self._model, name)[flat_ids] = source.repeat_interleave(self.num_entities)

    def reset(
        self,
        env_ids: torch.Tensor,
        position_w: torch.Tensor,
        velocity_w: torch.Tensor,
        interceptor_model,
    ) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        rows = env_ids.numel()
        flat_ids = self._flat_ids(env_ids)
        position_flat = self._flatten_entities(position_w, rows)
        velocity_flat = self._flatten_entities(velocity_w, rows)
        self._model.reset(
            flat_ids,
            position_flat,
            self._identity_quaternion[flat_ids],
            velocity_flat,
            self._zero_body_rate[flat_ids],
        )
        self.sync_timing(interceptor_model, env_ids)

    def capture(self, position_w: torch.Tensor, velocity_w: torch.Tensor) -> None:
        position_flat = self._flatten_entities(position_w, self.num_envs)
        velocity_flat = self._flatten_entities(velocity_w, self.num_envs)
        self._model.capture(
            position_flat,
            self._identity_quaternion,
            velocity_flat,
            self._zero_body_rate,
        )

    def sample(
        self,
        env_origins: torch.Tensor,
        interceptor_model,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.sync_timing(interceptor_model)
        expanded_origins = env_origins.repeat_interleave(self.num_entities, dim=0)
        _, position_local, velocity_w, _ = self._model.sample(expanded_origins)
        entity_shape = (self.num_envs, self.num_entities, 3)
        return position_local.reshape(entity_shape), velocity_w.reshape(entity_shape)
