# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import mujoco
import numpy as np
from judo import MODEL_PATH as JUDO_MODEL_PATH
from judo.tasks.base import TaskConfig
from judo.tasks.spot.spot_base import SpotBase as _JudoSpotBase
from judo.tasks.spot.spot_base import SpotBaseConfig

from sumo import MODEL_PATH

XML_PATH = str(JUDO_MODEL_PATH / "xml" / "spot_primitive" / "robot.xml")

ConfigT = TypeVar("ConfigT", bound=TaskConfig)

_INCLUDE_RE = re.compile(r'(<include\s+file=")([^"]+)(")')
_SPOT_PRIMITIVE_FILES = {
    "default.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "default.xml",
    "assets.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "assets.xml",
    "body.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "body.xml",
    "legs.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "legs.xml",
    "arm.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "arm.xml",
    "actuator.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "actuator.xml",
    "contact.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "contact.xml",
}
_SUMO_SPOT_EXTENSION_FILES = {
    "sensor.xml": MODEL_PATH / "xml" / "spot_components" / "sensor.xml",
}
_PUBLIC_OBJECT_COMPAT = {
    "tire_rubber.xml": JUDO_MODEL_PATH / "xml" / "objects" / "tire" / "tire.xml",
    "tire_rubber_defs.xml": JUDO_MODEL_PATH / "xml" / "objects" / "tire" / "tire_defs.xml",
}


def _get_spot_menagerie_dir() -> Path:
    try:
        from robot_descriptions import spot_mj_description  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - exercised via failure-path test
        raise RuntimeError(
            "Spot tasks require `robot_descriptions` to resolve Spot robot assets. "
            "Run from the root pixi environment (for example `pixi run sumo`) "
            "or install `robot_descriptions` in the active environment."
        ) from exc

    return Path(spot_mj_description.PACKAGE_PATH)


def _resolve_public_object_asset(relpath: str) -> Path | None:
    if "objects/" not in relpath:
        return None

    suffix = relpath[relpath.rindex("objects/") :]
    candidates = [suffix]
    if suffix.startswith("objects/tire/meshes/"):
        compat_suffix = suffix.replace("objects/tire/meshes/", "objects/tire/", 1)
        if compat_suffix.endswith("/visual/tire.obj"):
            compat_suffix = compat_suffix[: -len("/visual/tire.obj")] + "/visual/tire_rubber.obj"
        candidates.insert(0, compat_suffix)

    for root in (JUDO_MODEL_PATH / "meshes", MODEL_PATH / "meshes"):
        for candidate_suffix in candidates:
            candidate = root / candidate_suffix
            if candidate.exists():
                return candidate
    return None


# Name of the virtual camera mounted on Spot's head (front of the chassis).
SPOT_HEAD_CAMERA_NAME = "spot_head"


class SpotAssetMixin:
    """Align Spot robot assets with judo while preserving local object compatibility."""

    spec: Any  # MjSpec, provided by SpotBase via MRO

    def _process_spec(self) -> None:
        menagerie_dir = _get_spot_menagerie_dir()
        menagerie_assets = menagerie_dir / "assets"

        for mesh in self.spec.meshes:
            if "spot/meshes/" in mesh.file:
                mesh.file = str(menagerie_assets / Path(mesh.file).name)
                continue

            asset_path = _resolve_public_object_asset(mesh.file)
            if asset_path is not None:
                mesh.file = str(asset_path)

        for texture in self.spec.textures:
            if "spot/textures/" in texture.file:
                texture.file = str(menagerie_dir / "spot.png")

        self._add_head_camera()

    def _add_head_camera(self) -> None:
        """Mount a forward-looking virtual camera on Spot's head (front of the chassis).

        The camera is attached to the ``body`` body so it tracks the robot as it moves.
        It looks along the body's +x axis (forward) with +z (world up) as the image-up
        direction, mimicking a head-mounted RGB camera. A small visual-only marker
        (no collision) is added so the camera's mount and view direction are visible in
        the 3D scene.
        """
        if any(cam.name == SPOT_HEAD_CAMERA_NAME for cam in self.spec.cameras):
            return

        try:
            body = self.spec.body("body")
        except (ValueError, KeyError):
            # No chassis body to attach to (non-standard Spot model); skip silently.
            return

        # Front-top of the chassis (body collision box half-extents are 0.42 x 0.11 x 0.08).
        cam_pos = np.array([0.44, 0.0, 0.10])

        cam = body.add_camera()
        cam.name = SPOT_HEAD_CAMERA_NAME
        cam.pos = cam_pos
        cam.fovy = 90.0
        # Look forward (body +x) with world-up (+z) as image up. A small downward pitch
        # can be dialed in via ``pitch_down`` if the target sits low in frame, but 0deg
        # keeps distant targets (near the horizon) in view. The columns of the rotation
        # matrix are the camera x/y/z axes in the body frame (camera looks along -z):
        #   x_cam (right) = (0, -1, 0)
        #   y_cam (up)    = (sin a, 0, cos a)
        #   z_cam (-look) = (-cos a, 0, sin a)
        pitch_down = np.deg2rad(0.0)
        s, c = np.sin(pitch_down), np.cos(pitch_down)
        rot = np.array([0.0, s, -c, -1.0, 0.0, 0.0, 0.0, c, s])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot)
        cam.quat = quat

        self._add_head_camera_marker(body, cam_pos, quat, look=np.array([c, 0.0, -s]))

    def _add_head_camera_marker(self, body: Any, cam_pos: np.ndarray, cam_quat: np.ndarray, look: np.ndarray) -> None:
        """Add a visual-only marker showing the head camera's mount and view direction.

        The marker geoms carry no collision (contype/conaffinity = 0) and live in geom
        group 3, which MuJoCo's renderer hides by default (so the marker never appears in
        the camera's own image), while viser renders all groups (so it shows in the GUI).
        """
        # Camera housing: a small box aligned with the camera frame at the mount point.
        housing = body.add_geom()
        housing.name = "spot_head_cam_housing"
        housing.type = mujoco.mjtGeom.mjGEOM_BOX
        housing.pos = cam_pos
        housing.quat = cam_quat
        housing.size = [0.025, 0.035, 0.02]
        housing.rgba = [0.12, 0.12, 0.14, 1.0]
        housing.contype = 0
        housing.conaffinity = 0
        housing.group = 3

        # View direction: a "lens" cylinder pointing along the camera's look direction.
        lens = body.add_geom()
        lens.name = "spot_head_cam_lens"
        lens.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        tip = cam_pos + look * 0.12
        lens.fromto = [*cam_pos.tolist(), *tip.tolist()]
        lens.size = [0.01, 0.0, 0.0]  # radius; length derives from fromto
        lens.rgba = [0.95, 0.35, 0.05, 1.0]
        lens.contype = 0
        lens.conaffinity = 0
        lens.group = 3


class YawFloorMixin:
    """Judo's command mapping plus the yaw-rate command floor (`yaw_command_floor`).

    Applied on the way into the 25-dim policy command, so the planner's rollouts and the
    deployed policy node (which calls task_to_sim_ctrl per tick) execute the same
    command. Put FIRST in a task's bases, ahead of the judo task class.
    """

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        out = super().task_to_sim_ctrl(controls)  # type: ignore[misc]
        cfg = getattr(self, "config", None)
        floor = float(getattr(cfg, "yaw_rate_min", 0.0) or 0.0)
        if floor > 0.0:
            dead = float(getattr(cfg, "yaw_rate_deadzone", 0.0) or 0.0)
            # Never above the task's own hard yaw bound (spot_barrel_look_at narrows it
            # to max_yaw_rate): a floor is a floor, not a way past the cap.
            try:
                cap = float(np.max(np.abs(np.asarray(self.actuator_ctrlrange)[2])))  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 -- no bounds: no cap
                cap = float("inf")
            out[..., 2] = yaw_command_floor(out[..., 2], min(floor, cap), dead)
        return out


class SpotBase(SpotAssetMixin, YawFloorMixin, _JudoSpotBase, Generic[ConfigT]):
    """Sumo SpotBase wrapper that composes local task XML with public Spot definitions."""

    config_t: type[ConfigT]  # pyright: ignore[reportIncompatibleVariableOverride]
    config: ConfigT

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    @classmethod
    def _resolve_include_path(cls, include_path: str, source_dir: Path) -> Path:
        include = Path(include_path)
        basename = include.name

        if basename in _SPOT_PRIMITIVE_FILES:
            return _SPOT_PRIMITIVE_FILES[basename]
        if basename in _SUMO_SPOT_EXTENSION_FILES:
            return _SUMO_SPOT_EXTENSION_FILES[basename]
        if basename in _PUBLIC_OBJECT_COMPAT:
            return _PUBLIC_OBJECT_COMPAT[basename]

        candidate = include if include.is_absolute() else (source_dir / include).resolve()
        if candidate.exists():
            return candidate

        flat_object_candidate = MODEL_PATH / "xml" / "objects" / basename
        if flat_object_candidate.exists():
            return flat_object_candidate

        nested_object_candidate = MODEL_PATH / "xml" / "objects" / include.stem / basename
        if nested_object_candidate.exists():
            return nested_object_candidate

        raise FileNotFoundError(f"Unable to resolve include '{include_path}' from '{source_dir}'.")

    @classmethod
    def _materialize_model_path(cls, model_path: str | Path) -> Path:
        path = Path(model_path).expanduser().resolve()
        if cls._is_relative_to(path, JUDO_MODEL_PATH):
            return path

        source = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S)

        def replace_include(match: re.Match[str]) -> str:
            resolved = cls._resolve_include_path(match.group(2), path.parent)
            return f"{match.group(1)}{resolved.as_posix()}{match.group(3)}"

        rewritten = _INCLUDE_RE.sub(replace_include, source)
        cache_dir = Path(tempfile.gettempdir()) / "sumo-spot-models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(rewritten.encode("utf-8")).hexdigest()[:12]
        materialized_path = cache_dir / f"{path.stem}-{digest}.xml"
        if not materialized_path.exists():
            materialized_path.write_text(rewritten)
        return materialized_path

    def __init__(
        self,
        model_path: str = XML_PATH,
        use_arm: bool = True,
        use_gripper: bool = False,
        use_legs: bool = False,
        use_torso: bool = False,
        config: ConfigT | None = None,
    ) -> None:
        super().__init__(
            model_path=str(self._materialize_model_path(model_path)),
            use_arm=use_arm,
            use_gripper=use_gripper,
            use_legs=use_legs,
            use_torso=use_torso,
            config=cast(Any, config),
        )


def yaw_command_floor(wz: np.ndarray, floor: float, deadzone: float) -> np.ndarray:
    """|wz| below `deadzone` -> 0; otherwise |wz| is raised to at least `floor`.

    The Spot locomotion policy does not turn for small yaw-rate commands: on the real
    robot (2026-09-14) -0.13..-0.21 rad/s produced 0 deg/s and only ~0.4 rad/s turned it;
    MuJoCo with the same policy behaves alike. A sampling planner does not see that
    cliff as a cliff: it settles on a cheap small command that does nothing. With the
    floor, the command the policy receives is either "do not turn" or "turn at a rate
    that actually turns", in the rollouts and on the robot alike, and the planner's
    choice among those is honest. Off (returned unchanged) when floor <= 0.
    """
    wz = np.asarray(wz, dtype=float)
    if floor <= 0.0:
        return wz
    mag = np.abs(wz)
    return np.where(mag < deadzone, 0.0, np.sign(wz) * np.maximum(mag, floor))


__all__ = ["SPOT_HEAD_CAMERA_NAME", "SpotAssetMixin", "SpotBase", "SpotBaseConfig", "YawFloorMixin",
           "yaw_command_floor"]
