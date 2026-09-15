import mujoco
import numpy as np
import pytest

from sumo.tasks.spot import jug_neck_grasp
from sumo.tasks.spot.spot_jug_manipulation import SpotJugUpright, SpotJugUprightConfig, disable_velocity_rewards


@pytest.fixture(scope="module")
def task():
    config = SpotJugUprightConfig(neck_grasp=True, water_fill_ratio=0.1, jug_mass=1.0)
    disable_velocity_rewards(config)
    return SpotJugUpright(config)


def test_grasp_geometry_preserves_mass_and_state_size(task):
    assert task.model.nq == 166
    assert task.model.body("jug").mass[0] == 1
    assert task.model.geom("jug_neck_collision").size[0] == pytest.approx(0.029)
    assert task.model.geom("jug_neck_collision").contype[0] == 1
    assert task.model.geom("jug_water_0_geom").contype[0] == 2
    task.reset()
    assert task.metrics(task.data)["water_retained_fraction"] == 1
    assert not task.metrics(task.data)["neck_bilateral_contact"]


def test_grasp_requires_both_jaws_and_correct_closing_sign(task):
    states = np.tile(np.r_[task.data.qpos, task.data.qvel], (4, 3, 1))
    sensors = np.tile(task.data.sensordata, (4, 3, 1))
    controls = np.zeros((4, 3, task.nu))
    i = task.get_sensor_start_index("jug_pinch_local")
    sensors[..., i : i + 3] = jug_neck_grasp.TARGET
    for name in jug_neck_grasp.JAW_GEOMS:
        sensors[..., task.get_sensor_start_index(f"jug_neck_distance_{name}")] = -0.001
    states[..., task.get_joint_position_start_index("arm_f1x")] = -0.3
    controls[..., 10] = 1
    controls[1, :, 9] = -1.0  # Opening is NOT resistance to closing.
    for name in jug_neck_grasp.JAW_GEOMS[2:]:
        sensors[2, :, task.get_sensor_start_index(f"jug_neck_distance_{name}")] = 0.05
    states[3, :, task.get_joint_position_start_index("arm_f1x")] = 0  # Empty close.
    grasp = jug_neck_grasp.grasp_quantities(task, states, sensors, controls)[-1]
    np.testing.assert_array_equal(grasp[:, 0], [True, False, False, False])
    # Selection <0 actually commands closed even if raw finger command is open.
    controls[0, :, 9] = -1
    controls[0, :, 10] = -1
    assert jug_neck_grasp.grasp_quantities(task, states, sensors, controls)[-1][0].all()


def test_grasp_reward_shape_and_no_velocity_dependency(task):
    states = np.tile(np.r_[task.data.qpos, task.data.qvel], (3, 4, 1))
    sensors = np.tile(task.data.sensordata, (3, 4, 1))
    controls = np.zeros((3, 4, task.nu))
    reward = task.reward(states, sensors, controls)
    assert reward.shape == (3,) and np.isfinite(reward).all()
    states[..., task.model.nq :] = 10
    np.testing.assert_array_equal(task.reward(states, sensors, controls), reward)
    data = mujoco.MjData(task.model)
    data.qpos[:] = task.reset_pose
    mujoco.mj_forward(task.model, data)
    report = jug_neck_grasp.grasp_metrics(task, data, controls[0, 0])
    assert not report["neck_grasp_detected"]


def test_grasp_bonus_does_not_reward_remaining_partly_upright(task):
    states = np.tile(np.r_[task.data.qpos, task.data.qvel], (3, 1, 1))
    sensors = np.tile(task.data.sensordata, (3, 1, 1))
    controls = np.zeros((3, 1, task.nu))
    i = task.get_sensor_start_index("jug_pinch_local")
    sensors[..., i : i + 3] = jug_neck_grasp.TARGET
    for name in jug_neck_grasp.JAW_GEOMS:
        sensors[..., task.get_sensor_start_index(f"jug_neck_distance_{name}")] = -0.001
    states[..., task.get_joint_position_start_index("arm_f1x")] = -0.5
    cosine = np.array([0.75, 0.9, 0.99])
    sensors[:, 0, task.object_z_axis_start + 2] = cosine
    terms = task.reward_terms(states, sensors, controls)
    assert np.all(np.diff(terms["orientation"] + terms["neck_grasp"]) > 0)


@pytest.mark.parametrize("width", [0, -1, np.nan, np.inf])
def test_invalid_grasp_fade_width(width):
    with pytest.raises(ValueError, match="neck_grasp_fade_width"):
        SpotJugUpright(SpotJugUprightConfig(neck_grasp=True, neck_grasp_fade_width=width))


def test_upright_grasp_task_is_registered_with_the_grasp_geometry():
    from judo.tasks import get_registered_tasks

    import sumo.tasks  # noqa: F401
    from sumo.tasks.spot.spot_jug_manipulation import SpotJugUpright, SpotJugUprightGrasp

    reg = get_registered_tasks()
    assert "spot_jug_upright_grasp" in reg
    grasp = SpotJugUprightGrasp()
    plain = SpotJugUpright()
    assert grasp.neck_grasp and not plain.neck_grasp
    assert grasp.nu == plain.nu == 11
    assert "grasp" in "".join(grasp.reward_terms(*_zero_inputs(grasp)).keys())


def _zero_inputs(task):
    import numpy as np

    s = np.zeros((1, 1, task.model.nq + task.model.nv))
    s[..., : task.model.nq] = task.reset_pose[: task.model.nq]
    return s, np.zeros((1, 1, task.model.nsensordata)), np.zeros((1, 1, task.nu))
