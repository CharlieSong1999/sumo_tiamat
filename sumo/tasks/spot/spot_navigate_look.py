"""spot_navigate_look: judo's spot_navigate plus the operator look-at point.

No perceived objects, nu=3 base velocity -- the task to switch to when a perceived
object is lost and the planner would otherwise sit idle: the operator keeps a
destination (goal_pos, the monitor's "move" click) and a heading target
(look_at_pos, the "look at" click) that are set independently.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.tasks.spot.spot_navigate import SpotNavigate, SpotNavigateConfig

from sumo.tasks.spot.look_at import LookAtPointFields, point_look_term


@dataclass
class SpotNavigateLookConfig(LookAtPointFields, SpotNavigateConfig):
    pass


class SpotNavigateLook(SpotNavigate):
    name = "spot_navigate_look"
    config_t: type[SpotNavigateLookConfig] = SpotNavigateLookConfig  # type: ignore[assignment]
    config: SpotNavigateLookConfig

    def __init__(self, config: "SpotNavigateLookConfig | None" = None) -> None:
        super().__init__(config=config)

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: "dict[str, Any] | None" = None,
    ) -> np.ndarray:
        base = super().reward(states, sensors, controls, system_metadata)
        look = point_look_term(states[..., : self.model.nq], self.body_pose_idx, self.config)
        assert look.shape == base.shape
        return base + look
