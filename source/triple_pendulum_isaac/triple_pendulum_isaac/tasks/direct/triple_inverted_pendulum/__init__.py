"""Gym registration for staged triple inverted pendulum tasks."""

import gymnasium as gym

from . import agents

gym.register(
    id="TriplePendulum-Balance-Direct-v0",
    entry_point=f"{__name__}.env:TripleInvertedPendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:TripleInvertedPendulumBalanceEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="TriplePendulum-SwingUp-Direct-v0",
    entry_point=f"{__name__}.env:TripleInvertedPendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:TripleInvertedPendulumSwingUpEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="TriplePendulum-SwingUpBalance-Direct-v0",
    entry_point=f"{__name__}.env:TripleInvertedPendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:TripleInvertedPendulumSwingUpBalanceEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
