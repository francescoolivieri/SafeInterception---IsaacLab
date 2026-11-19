"""DG-PPO agent configuration (the agent block of skrl_dgppo_cfg.yaml)."""
import dataclasses
from collections.abc import Mapping
from typing import Any

from skrl.agents.torch import AgentCfg
from skrl.agents.torch.base import ExperimentCfg

from target_interception.nn.networks import resolve_graph_input, resolve_networks

# Upstream DG-PPO options, supported only at these values (saved configs list them).
FIXED = {
    "bootstrap_on_truncated": False,
    "use_rnn": False,
    "cbf_schedule": True,
    "policy_gradient_mode": "approximate",
    "rewards_shaper_scale": 1,
    "learning_starts": 0,
}
IGNORED = {"rnn", "rnn_step", "deterministic", "debug_rollout_plot_interval"}


@dataclasses.dataclass(kw_only=True)
class DGPPOAgentCfg(AgentCfg):
    alpha: float = 2.0  # class-K gain of the discrete CBF condition
    cbf_eps: float = 1e-2
    cbf_weight: float = 1.0
    # After the curriculum, the CBF weight is multiplied stage by stage; each fraction is
    # the share of the remaining steps that its multiplier is active.
    cbf_schedule_fractions: tuple[float, ...] = (0.01, 0.5, 0.49)
    cbf_schedule_multipliers: tuple[float, ...] = (1.0, 2.0, 4.0)

    discount_factor: float = 0.995
    gae_lambda: float = 0.95
    rollouts: int = 32
    learning_epochs: int = 4
    mini_batches: int = 4
    ratio_clip: float = 0.25
    entropy_loss_scale: float = 0.003
    vl_loss_scale: float = 1.0
    vh_loss_scale: float = 1.0
    grad_norm_clip: float = 0.8
    lr_policy: float = 3e-4
    lr_vl: float = 3e-4
    lr_vh: float = 3e-4

    # Resolved specs from task/agents/networks.yaml.
    graph_input: dict[str, Any] | str | None = None
    networks: dict[str, Any] = dataclasses.field(default_factory=dict)  # policy, vl, vh
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DGPPOAgentCfg":
        raw = dict(data or {})
        for key, value in FIXED.items():
            if key in raw and raw.pop(key) != value:
                raise ValueError(f"DG-PPO only implements {key}={value!r}")
        for key in IGNORED:
            raw.pop(key, None)
        experiment = raw.pop("experiment", {})
        cfg = cls(**raw)
        cfg.experiment = experiment if isinstance(experiment, ExperimentCfg) else ExperimentCfg(**experiment)
        cfg.graph_input = resolve_graph_input(cfg.graph_input)
        cfg.networks = resolve_networks(cfg.networks)
        if set(cfg.networks) != {"policy", "vl", "vh"}:
            raise ValueError(f"DG-PPO needs the policy, vl and vh networks, got {sorted(cfg.networks)}")
        cfg.cbf_schedule_fractions = tuple(float(v) for v in cfg.cbf_schedule_fractions)
        cfg.cbf_schedule_multipliers = tuple(float(v) for v in cfg.cbf_schedule_multipliers)
        if len(cfg.cbf_schedule_fractions) != len(cfg.cbf_schedule_multipliers):
            raise ValueError("cbf_schedule_fractions and cbf_schedule_multipliers must have equal length")
        return cfg
