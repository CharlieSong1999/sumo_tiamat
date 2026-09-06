import mujoco
import numpy as np
import pytest
from judo.tasks import get_registered_tasks

from sumo.tasks.spot.spot_barrel_perceive import BARREL_REST_HEIGHT, SpotBarrelPerceive


@pytest.fixture(scope="module")
def task():
    return SpotBarrelPerceive()


def test_registered():
    assert "spot_barrel_perceive" in get_registered_tasks()


def test_deployment_morphology(task):
    """Locomotion-only, same action space as the deployed spot_navigate."""
    assert task.nu == 3
    assert not task.use_arm


def test_perceived_barrel_object(task):
    assert task.perceived_object_joints == ("barrel_joint",)
    assert task.model.joint("barrel_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert len(task.reset_pose) == task.model.nq


def test_barrel_rests_upright_ahead(task):
    o = task.get_joint_position_start_index("barrel_joint")
    pose = np.array(task.reset_pose)[o : o + 7]
    assert pose[2] == pytest.approx(BARREL_REST_HEIGHT)
    np.testing.assert_allclose(pose[3:], [1, 0, 0, 0])  # upright before first measurement


def test_reward_shape(task):
    batch, horizon = 4, 6
    rng = np.random.default_rng(0)
    states = rng.normal(size=(batch, horizon, task.model.nq + task.model.nv))
    sensors = rng.normal(size=(batch, horizon, task.model.nsensordata))
    controls = rng.normal(size=(batch, horizon, task.nu))
    assert task.reward(states, sensors, controls).shape == (batch,)
