from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from leisaac.utils.robot_utils import is_so101_at_rest_pose

from .observations import cube_placed_on_correct_target


def cube_height_above_base(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    robot_base_name: str = "base",
    height_threshold: float = 0.20,
) -> torch.Tensor:
    """Determine if the cube is above the robot base.

    This function checks whether all success conditions for the task have been met:
    1. cube is above the robot base

    Args:
        env: The RL environment instance.
        cube_cfg: Configuration for the cube entity.
        robot_cfg: Configuration for the robot entity.
        robot_base_name: Name of the robot base.
        height_threshold: Threshold for the cube height above the robot base.
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_height = cube.data.root_pos_w[:, 2]
    base_index = robot.data.body_names.index(robot_base_name)
    robot_base_height = robot.data.body_pos_w[:, base_index, 2]
    above_base = cube_height - robot_base_height > height_threshold
    done = torch.logical_and(done, above_base)

    return done


def cube_pick_place_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    circle_target_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    center_x: float = 0.0,
) -> torch.Tensor:
    """Check that the cube was placed on its correct target (by spawn side) and the SO-101 returned to rest."""
    placed = cube_placed_on_correct_target(
        env,
        cube_cfg=cube_cfg,
        target_cfg=target_cfg,
        circle_target_cfg=circle_target_cfg,
        robot_cfg=robot_cfg,
        center_x=center_x,
    )
    robot: Articulation = env.scene[robot_cfg.name]
    at_rest = is_so101_at_rest_pose(robot.data.joint_pos, robot.data.joint_names)
    return torch.logical_and(placed, at_rest)
