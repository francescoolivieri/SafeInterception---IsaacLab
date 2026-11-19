"""skrl models for graph PPO (M4): the DG-PPO graph policy and a scalar graph critic."""
from __future__ import annotations

from typing import Any

import torch
from skrl.models.torch import DeterministicMixin, Model

from target_interception.nn.networks import GraphPolicy, GraphValueNet
from target_interception.nn.graph import GraphData, RelationGraph


class GraphObservationEncoder:
    """Build and preprocess one graph, shared by the actor and critic calls on the same batch."""

    def __init__(self, graph: RelationGraph, feature_preprocessor) -> None:
        self.graph = graph
        self.feature_preprocessor = feature_preprocessor
        self._cached_key: tuple | None = None
        self._cached_graph: GraphData | None = None

    def __call__(self, observations: torch.Tensor) -> GraphData:
        observations = observations.reshape(-1, observations.shape[-1])
        # Inference tensors have no version counter, so an in-place update of reused
        # observation storage could not be detected: never cache them.
        key = None
        if not torch.is_inference(observations):
            key = (observations.data_ptr(), observations.storage_offset(), tuple(observations.shape),
                   tuple(observations.stride()), observations._version)
            if key == self._cached_key and self._cached_graph is not None:
                return self._cached_graph
        graph = self.feature_preprocessor(self.graph(observations))
        self._cached_key, self._cached_graph = key, (graph if key is not None else None)
        return graph

    def clear_cache(self) -> None:
        self._cached_key = None
        self._cached_graph = None

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        """Update the feature statistics after PPO has finished with a rollout."""
        self.feature_preprocessor.update(self.graph(observations.reshape(-1, observations.shape[-1])))
        self.clear_cache()


class GraphPPOPolicy(Model):
    """DG-PPO graph policy behind skrl's model API."""

    def __init__(
        self,
        *,
        core: GraphPolicy,
        encoder: GraphObservationEncoder,
        observation_space,
        state_space,
        action_space,
        device,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        self.core = core
        # Keep the shared, non-module graph builder out of the module tree. The
        # scaler is checkpointed once by GraphPPO rather than duplicated here.
        object.__setattr__(self, "_graph_encoder", encoder)
        self._distribution = None

    def _make_distribution(self, inputs: dict[str, Any]):
        distribution = self.core.distribution(self._graph_encoder(inputs["observations"]))
        self._distribution = distribution
        return distribution

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        distribution = self._make_distribution(inputs)
        mean_actions = distribution.mode().reshape(-1, self.num_actions)
        log_std = distribution.std.reshape(-1, self.num_actions).log()
        return mean_actions, {"log_std": log_std}

    def act(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        distribution = self._make_distribution(inputs)
        mean_actions = distribution.mode().reshape(-1, self.num_actions)
        actions = distribution.sample().reshape(-1, self.num_actions)
        evaluated_actions = inputs.get("taken_actions", actions).reshape(-1, self.num_actions)
        log_prob = distribution.log_prob(
            evaluated_actions.reshape(distribution.mean.shape)
        ).reshape(-1, 1)
        return actions, {
            "log_prob": log_prob,
            "mean_actions": mean_actions,
            "log_std": distribution.std.reshape(-1, self.num_actions).log(),
        }

    def get_entropy(self, role: str = "") -> torch.Tensor:
        if self._distribution is None:
            return torch.zeros(1, 1, device=self.device)
        return self._distribution.entropy().reshape(-1, 1)

    def distribution(self, role: str = ""):
        # skrl only uses this accessor to report the policy standard deviation.
        return None if self._distribution is None else self._distribution.normal


class GraphPPOValue(DeterministicMixin, Model):
    """Graph critic with one scalar value."""

    def __init__(
        self,
        *,
        core: GraphValueNet,
        encoder: GraphObservationEncoder,
        observation_space,
        state_space,
        action_space,
        device,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.core = core
        object.__setattr__(self, "_graph_encoder", encoder)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.core(self._graph_encoder(inputs["observations"])).reshape(-1, 1), {}
