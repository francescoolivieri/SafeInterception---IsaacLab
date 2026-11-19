"""Target-interception task registrations."""

import gymnasium as gym

from . import agents

_ENTRY = f"{__name__}.target_interception_env:TargetInterceptionEnv"
_CFG = f"{__name__}.target_interception_env_cfg"
_AGENTS = agents.__name__

# (task suffix, env cfg class, agent cfgs). The room task is used for out-of-distribution evaluation.
_TASKS = (
    ("State", "TargetInterceptionStateCfg", ()),
    ("State-Large", "TargetInterceptionStateLargeCfg", ("skrl_dgppo", "skrl_pncbf")),
    ("State-Penalty-Large", "TargetInterceptionStatePenaltyLargeCfg", ("skrl_ppo_graph",)),
    ("Nominal-Large", "TargetInterceptionNominalLargeCfg", ("skrl",)),
)

for suffix, cfg_name, agent_cfgs in _TASKS:
    kwargs = {"env_cfg_entry_point": f"{_CFG}:{cfg_name}"}
    for name in agent_cfgs:
        yaml_name = "skrl_ppo_cfg.yaml" if name == "skrl" else f"{name}_cfg.yaml"
        kwargs[f"{name}_cfg_entry_point"] = f"{_AGENTS}:{yaml_name}"
    gym.register(id=f"TargetInterception-{suffix}-v0", entry_point=_ENTRY, disable_env_checker=True, kwargs=kwargs)
