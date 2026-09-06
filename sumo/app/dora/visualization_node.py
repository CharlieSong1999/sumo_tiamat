# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
from dora_utils.node import on_event
from judo.app.dora.visualization_node import VisualizationNode as JudoVisualizationNode
from viser import GuiImageHandle

import sumo.controller  # noqa: F401 -- register controller/optimizer overrides
import sumo.tasks  # noqa: F401 -- register all sumo tasks
from sumo.tasks import get_sumo_registered_tasks


class VisualizationNode(JudoVisualizationNode):
    """Visualization node with sumo-only task choices and a head-camera viewport."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("available_tasks", get_sumo_registered_tasks())
        # Kept outside visualizer.gui_elements so it survives task switches (which
        # rebuild the rest of the GUI). A single persistent image panel is reused.
        self._cam_handle: GuiImageHandle | None = None
        super().__init__(**kwargs)

    @on_event("INPUT", "camera_image")
    def update_camera_image(self, event: dict) -> None:
        """Display the latest Spot head-camera frame in a viser image panel."""
        shape = tuple(int(s) for s in event["metadata"]["shape"])
        img = event["value"].to_numpy().reshape(shape)
        with self.visualizer.task_lock:
            if self._cam_handle is None:
                self._cam_handle = self.visualizer.server.gui.add_image(
                    np.ascontiguousarray(img),
                    label="Head Camera",
                )
            else:
                self._cam_handle.image = np.ascontiguousarray(img)


__all__ = ["VisualizationNode"]
