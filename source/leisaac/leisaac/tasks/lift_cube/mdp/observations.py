import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def object_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.02,
    grasp_threshold: float = 0.26,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)

    return grasped


def cube_placed_on_target(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
    height_range: tuple[float, float] = (0.0, 0.06),
    grasp_threshold: float = 0.26,
    speed_threshold: float = 0.10,
) -> torch.Tensor:
    """Check that the released cube is stationary on the target region."""
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    position_delta = cube.data.root_pos_w - target.data.root_pos_w
    in_target = torch.logical_and(position_delta[:, 0] > x_range[0], position_delta[:, 0] < x_range[1])
    in_target = torch.logical_and(in_target, position_delta[:, 1] > y_range[0])
    in_target = torch.logical_and(in_target, position_delta[:, 1] < y_range[1])
    in_target = torch.logical_and(in_target, position_delta[:, 2] > height_range[0])
    in_target = torch.logical_and(in_target, position_delta[:, 2] < height_range[1])

    gripper_open = robot.data.joint_pos[:, -1] > grasp_threshold
    cube_stationary = torch.linalg.vector_norm(cube.data.root_lin_vel_w, dim=1) < speed_threshold
    return torch.logical_and(torch.logical_and(in_target, gripper_open), cube_stationary)
