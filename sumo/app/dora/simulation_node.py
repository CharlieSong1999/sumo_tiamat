# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import os

# Select a headless OpenGL backend for MuJoCo offscreen rendering BEFORE mujoco is
# imported anywhere in this process. EGL works on machines with an NVIDIA/Mesa GPU
# (including over SSH); override with MUJOCO_GL=osmesa for pure-CPU rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

import warnings

import mujoco
import numpy as np
import pyarrow as pa
from judo.app.dora.simulation_node import SimulationNode as JudoSimulationNode

import sumo.tasks  # noqa: F401 -- register all sumo tasks
from sumo.app.dora.g1_simulation import G1Simulation
from sumo.tasks.spot.spot_base import SPOT_HEAD_CAMERA_NAME


class SimulationNode(JudoSimulationNode):
    """Simulation node with G1 backend support and a virtual head camera.

    In addition to the standard state/render outputs, this node renders the Spot
    head camera offscreen and publishes the RGB frame on the ``camera_image`` topic
    for the visualization node to display.
    """

    def __init__(
        self,
        init_task: str = "spot_box_push",
        camera_name: str = SPOT_HEAD_CAMERA_NAME,
        camera_width: int = 320,
        camera_height: int = 240,
        camera_fps: float = 20.0,
        **kwargs,
    ) -> None:
        kwargs.setdefault("backend_registry", {"mujoco_g1": G1Simulation})

        self._camera_name = camera_name
        self._camera_width = int(camera_width)
        self._camera_height = int(camera_height)
        self._camera_fps = float(camera_fps)

        self._renderer: mujoco.Renderer | None = None
        self._rendered_model: mujoco.MjModel | None = None
        self._render_every = 1
        self._render_counter = 0

        # JudoSimulationNode.__init__ calls write_states() at the end, so the camera
        # attributes above must already be set before we delegate to super().
        super().__init__(init_task=init_task, **kwargs)

    def _ensure_renderer(self) -> None:
        """(Re)create the offscreen renderer when the active model changes."""
        model = self.sim.task.model
        if self._rendered_model is model:
            return

        # Model changed (startup or task switch): drop the stale renderer/context.
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._rendered_model = model

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name)
        if cam_id < 0:
            # Task has no head camera (e.g. G1 tasks): nothing to render.
            return

        try:
            self._renderer = mujoco.Renderer(model, self._camera_height, self._camera_width)
        except Exception as e:  # pragma: no cover - depends on GL backend availability
            warnings.warn(
                f"Could not create MuJoCo offscreen renderer ({e!r}); head camera disabled. "
                "Try MUJOCO_GL=osmesa for CPU rendering.",
                stacklevel=2,
            )
            self._renderer = None
            return

        dt = self.sim.timestep
        self._render_every = max(1, round((1.0 / self._camera_fps) / dt)) if dt > 0 else 1
        self._render_counter = 0

    def write_states(self) -> None:
        """Write states (base behavior) and additionally publish a head-camera frame."""
        super().write_states()
        self._publish_camera_image()

    def _publish_camera_image(self) -> None:
        """Render the head camera (throttled) and publish the RGB frame."""
        self._ensure_renderer()
        if self._renderer is None:
            return

        self._render_counter += 1
        if self._render_counter % self._render_every != 0:
            return

        try:
            self._renderer.update_scene(self.sim.task.data, camera=self._camera_name)
            img = self._renderer.render()  # (H, W, 3) uint8 RGB
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(f"Head-camera render failed: {e!r}", stacklevel=2)
            return

        img = np.ascontiguousarray(img, dtype=np.uint8)
        self.node.send_output(
            "camera_image",
            pa.array(img.reshape(-1)),
            metadata={"shape": img.shape},
        )

    def cleanup(self) -> None:
        """Release the offscreen renderer before shutting down."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().cleanup()


__all__ = ["G1Simulation", "SimulationNode"]
