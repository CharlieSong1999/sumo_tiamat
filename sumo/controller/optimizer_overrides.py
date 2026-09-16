# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.config import set_config_overrides
from judo.optimizers.base import OptimizerConfig
from judo.optimizers.cem import CrossEntropyMethodConfig
from judo.optimizers.mppi import MPPIConfig
from judo.optimizers.ps import PredictiveSamplingConfig

from sumo.tasks import SPOT_TASK_NAMES

_SPOT_OPTIMIZER_BASE = {
    "num_rollouts": 24,
    "num_nodes": 3,
    "use_noise_ramp": True,
    "noise_ramp": 3.5,
}


def _set_spot_optimizer_overrides(task_name: str) -> None:
    """Sets the default optimizer config overrides for a Spot task."""
    set_config_overrides(task_name, OptimizerConfig, _SPOT_OPTIMIZER_BASE)
    set_config_overrides(task_name, PredictiveSamplingConfig, _SPOT_OPTIMIZER_BASE)
    set_config_overrides(task_name, CrossEntropyMethodConfig, {**_SPOT_OPTIMIZER_BASE, "num_elites": 3})
    set_config_overrides(task_name, MPPIConfig, _SPOT_OPTIMIZER_BASE)


# spot_jug_upright drives the arm (nu=11) and did not converge at the deployed shape
# (3 knots, 3 elites, noise_ramp 3.5; rehearsal 2026-09-16). Offline it succeeds at the
# same 24 rollouts x 1 iteration with 2 elites (the 48x2 runs' top-k ratio), 4 knots,
# sigma_min 0.12 and noise_ramp 2 (docs/jug_upright_24x1.md). Same rollout count, so the
# per-plan cost stays that of the nu=3 family. tiamat's mpc_config declares the same
# shape and asserts it against the built controller.
_SPOT_UPRIGHT_CEM = {**_SPOT_OPTIMIZER_BASE, "num_nodes": 4, "num_elites": 2,
                     "sigma_min": 0.12, "noise_ramp": 2.0}


# The gentle arm-roll deployment profile (nu=11, 19 water balls): 32 rollouts x 1 iteration,
# 3 knots, 3 elites, sigma 0.1..1.0, noise_ramp 3.5 -- the shape of its authoritative
# 32x1 / 1.5 s run (10/12 stable arrivals offline vs 6/12 at 24x1 / 2 s). One third more
# rollouts than the nu=3 family; whether that fits the 50 ms plan budget on the deployed
# workstation is measured, not assumed (handoff §8.3). tiamat's mpc_config declares the
# same shape and asserts it against the built controller.
_SPOT_ROLL_ARM_GENTLE_CEM = {**_SPOT_OPTIMIZER_BASE, "num_rollouts": 32, "num_elites": 3,
                             "sigma_min": 0.1, "sigma_max": 1.0}


def set_default_spot_optimizer_overrides() -> None:
    """Sets the default task-specific optimizer config overrides for all Spot tasks."""
    for task_name in SPOT_TASK_NAMES:
        _set_spot_optimizer_overrides(task_name)
    for task_name in ("spot_jug_upright", "spot_jug_upright_grasp"):
        set_config_overrides(task_name, CrossEntropyMethodConfig, _SPOT_UPRIGHT_CEM)
    for task_name in ("spot_jug_roll_arm_gentle", "spot_jug_roll_arm_gentle_dry"):
        set_config_overrides(task_name, OptimizerConfig, {**_SPOT_OPTIMIZER_BASE, "num_rollouts": 32})
        set_config_overrides(task_name, CrossEntropyMethodConfig, _SPOT_ROLL_ARM_GENTLE_CEM)
