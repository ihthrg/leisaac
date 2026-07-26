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


def _cube_is_right_of_center(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    center_x: float,
) -> torch.Tensor:
    """Cache, per env, whether the cube spawned to the right (+X) of ``center_x``.

    The cube moves once it is grasped, so only envs that were just reset
    (``episode_length_buf <= 1``) are re-evaluated; the result is cached on ``env`` for the rest
    of the episode so later calls (e.g. after the cube has been picked up) still reflect the
    cube's original spawn side rather than its current, possibly-moved position.
    """
    if not hasattr(env, "_lift_cube_sort_is_right"):
        env._lift_cube_sort_is_right = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    if torch.any(just_reset):
        cube: RigidObject = env.scene[cube_cfg.name]
        current_is_right = cube.data.root_pos_w[:, 0] > center_x
        env._lift_cube_sort_is_right = torch.where(just_reset, current_is_right, env._lift_cube_sort_is_right)

    return env._lift_cube_sort_is_right


def cube_placed_on_correct_target(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    circle_target_cfg: SceneEntityCfg = SceneEntityCfg("circle_target"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    center_x: float = 0.0,
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
    height_range: tuple[float, float] = (0.0, 0.06),
    grasp_threshold: float = 0.26,
    speed_threshold: float = 0.10,
) -> torch.Tensor:
    """Check that the cube is placed on whichever target matches its original spawn side.

    If the cube spawned to the right (+X) of ``center_x`` it must be placed on
    ``circle_target_cfg``; otherwise it must be placed on ``target_cfg`` (the square marker).
    """
    is_right = _cube_is_right_of_center(env, cube_cfg, center_x)

    kwargs = {
        "cube_cfg": cube_cfg,
        "robot_cfg": robot_cfg,
        "x_range": x_range,
        "y_range": y_range,
        "height_range": height_range,
        "grasp_threshold": grasp_threshold,
        "speed_threshold": speed_threshold,
    }
    on_square = cube_placed_on_target(env, target_cfg=target_cfg, **kwargs)
    on_circle = cube_placed_on_target(env, target_cfg=circle_target_cfg, **kwargs)
    return torch.where(is_right, on_circle, on_square)
