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
from sumo.tasks.spot.spot_base import YawFloorMixin


@dataclass
class SpotNavigateLookConfig(LookAtPointFields, SpotNavigateConfig):
    # Flat-bottomed goal well and a control cost, as in spot_barrel_perceive: this is the
    # task the robot waits in, and waiting must be a zero command, not a 0.1 m/s shuffle.
    goal_tolerance: float = 0.10
    w_controls: float = 10.0   # on (vx, vy) only; yaw is handled by the command deadbands


class SpotNavigateLook(YawFloorMixin, SpotNavigate):
    name = "spot_navigate_look"
    config_t: type[SpotNavigateLookConfig] = SpotNavigateLookConfig  # type: ignore[assignment]
    config: SpotNavigateLookConfig
    yaw_floor_enabled: bool = True   # steering task: floor always on (both sides agree statically)

    def __init__(self, config: "SpotNavigateLookConfig | None" = None) -> None:
        super().__init__(config=config)

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: "dict[str, Any] | None" = None,
    ) -> np.ndarray:
        # judo's SpotNavigate reward with the goal well (judo's own has no tolerance).
        c = self.config
        qpos = states[..., : self.model.nq]
        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        fallen = -c.fall_penalty * (body_height <= c.spot_fallen_threshold).any(axis=-1)
        goal_dist = np.linalg.norm(body_pos - np.asarray(c.goal_pos)[None, None], axis=-1)
        goal = -c.w_goal * np.maximum(goal_dist - c.goal_tolerance, 0.0).mean(-1)
        ctrl = -c.w_controls * np.linalg.norm(controls[..., :2], axis=-1).mean(-1)
        if not c.look_at_enabled:   # holding: yaw commands are noise, make them cost
            ctrl = ctrl - c.w_yaw_hold * np.abs(controls[..., 2]).mean(-1)
        look = point_look_term(qpos, self.body_pose_idx, c)
        assert look.shape == goal.shape == fallen.shape
        return fallen + goal + ctrl + look
