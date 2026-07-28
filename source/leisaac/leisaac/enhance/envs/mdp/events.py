from typing import Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera

NOMINAL_CAMERA_POSE_ATTR = "_leisaac_nominal_camera_pose"
"""Attribute cached on a camera asset holding its world pose before any randomization."""


def _nominal_camera_pose(
    asset: Camera, convention: Literal["opengl", "ros", "world"]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the camera world pose captured before any randomization was applied.

    The randomization event runs on *every* episode reset. Offsetting from the camera's current
    pose would compound the perturbations into a random walk that keeps drifting throughout a
    recording session, so the pose observed before the first perturbation is kept as the anchor.
    The first reset covers every environment, so the whole buffer can be captured in one go.

    Note that the anchor is expressed in world coordinates. Cameras parented under a robot link
    therefore stop following that link across resets if the robot's own base pose is randomized.
    """
    cache = getattr(asset, NOMINAL_CAMERA_POSE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(asset, NOMINAL_CAMERA_POSE_ATTR, cache)
    if convention not in cache:
        if convention == "ros":
            quat_w = asset.data.quat_w_ros
        elif convention == "opengl":
            quat_w = asset.data.quat_w_opengl
        else:
            quat_w = asset.data.quat_w_world
        cache[convention] = (asset.data.pos_w.clone(), quat_w.clone())
    return cache[convention]


def randomize_camera_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation uniformly within the given ranges.

    * It samples the camera position and rotation from the given ranges and adds them to the
      camera's nominal (pre-randomization) position and rotation, before setting them into the
      physics simulation. Offsetting from the nominal pose rather than the current one keeps
      repeated resets from accumulating into a drift.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    asset: Camera = env.scene[asset_cfg.name]

    nominal_pos_w, nominal_quat_w = _nominal_camera_pose(asset, convention)
    ori_pos_w = nominal_pos_w[env_ids]
    ori_quat_w = nominal_quat_w[env_ids]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # camera usually spawn with robot, so no need to add env_origins
    positions = ori_pos_w[:, 0:3] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)


def randomize_particle_object_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
):
    """Reset the particle object to a random position and rotation uniformly within the given ranges.

    * It samples the particle object position and rotation from the given ranges and adds them to the
      default particle object position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    particle_object = env.scene.particle_objects[asset_cfg.name]
    ori_world_pos, ori_world_quat = particle_object.get_world_poses()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    positions = ori_world_pos + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_world_quat, orientations_delta)

    particle_object.set_world_poses(positions, orientations)


def disable_rigid_body_gravity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """Disable gravity for specific bodies in an articulation.

    This function disables gravity for bodies specified in the asset_cfg.body_names.
    It uses modify_rigid_body_properties to set disable_gravity=True for the specified bodies.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply the change to.
        asset_cfg: Configuration specifying the asset and body names to disable gravity for.
                   Use body_names to specify which bodies to disable gravity (e.g., ".*arm.*" or ["shoulder", "elbow"]).
    """
    # Get the asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve body indices from body_names (already resolved by SceneEntityCfg)
    if asset_cfg.body_ids == slice(None):
        body_ids = list(range(asset.num_bodies))
    else:
        body_ids = asset_cfg.body_ids if isinstance(asset_cfg.body_ids, list) else [asset_cfg.body_ids]

    # Get link paths from the first environment (they follow the same pattern for all environments)
    link_paths = asset.root_physx_view.link_paths[0]

    # Disable gravity for each specified body
    for body_id in body_ids:
        if body_id >= len(link_paths):
            continue

        # Get the link path from first environment
        first_env_link_path = link_paths[body_id]

        # Convert to regex expression by replacing env_0 with env_.*
        link_path_expr = first_env_link_path.replace("/env_0/", "/env_.*/")

        # Resolve all matching prim paths and apply
        prim_paths = sim_utils.find_matching_prim_paths(link_path_expr)
        for prim_path in prim_paths:
            sim_utils.modify_rigid_body_properties(
                prim_path,
                sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            )
