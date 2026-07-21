import math
from typing import Literal

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.enhance.envs.manager_based_rl_digital_twin_env_cfg import (
    ManagerBasedRLDigitalTwinEnvCfg,
)
from leisaac.utils.camera import OpenCvCameraCfg
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp

U20CAM_WIDTH = 640
U20CAM_HEIGHT = 480
U20CAM_HORIZONTAL_FOV_DEG = 106.0
U20CAM_FOCAL_LENGTH = 2.8
U20CAM_FX = U20CAM_WIDTH / (2 * math.tan(math.radians(U20CAM_HORIZONTAL_FOV_DEG / 2)))
U20CAM_FY = U20CAM_FX
U20CAM_CX = U20CAM_WIDTH / 2
U20CAM_CY = U20CAM_HEIGHT / 2
# The manufacturer does not publish calibration coefficients; replace these with per-camera calibration results.
U20CAM_DISTORTION_COEFFICIENTS = (0.0,) * 8


def _u20cam_cfg(
    fx: float = U20CAM_FX,
    fy: float = U20CAM_FY,
    cx: float = U20CAM_CX,
    cy: float = U20CAM_CY,
    distortion_model: Literal["pinhole", "fisheye"] = "pinhole",
    distortion_coefficients: tuple[float, ...] = U20CAM_DISTORTION_COEFFICIENTS,
) -> OpenCvCameraCfg:
    return OpenCvCameraCfg(
        calibration_width=U20CAM_WIDTH,
        calibration_height=U20CAM_HEIGHT,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_model=distortion_model,
        distortion_coefficients=distortion_coefficients,
        focal_length=U20CAM_FOCAL_LENGTH,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=U20CAM_WIDTH * U20CAM_FOCAL_LENGTH / fx,
        vertical_aperture=U20CAM_HEIGHT * U20CAM_FOCAL_LENGTH / fy,
        clipping_range=(0.01, 50.0),
        lock_camera=True,
    )


@configclass
class LiftCubeSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the lift cube task."""

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.6, -0.75, 0.38), rot=(0.77337, 0.55078, -0.2374, -0.20537), convention="opengl"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.6,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"
        ),
        data_types=["rgb"],
        spawn=_u20cam_cfg(),
        width=U20CAM_WIDTH,
        height=U20CAM_HEIGHT,
        update_period=1 / 30.0,
    )

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.225, -0.5, 0.6), rot=(0.1650476, -0.9862856, 0.0, 0.0), convention="ros"
        ),
        data_types=["rgb"],
        spawn=_u20cam_cfg(),
        width=U20CAM_WIDTH,
        height=U20CAM_HEIGHT,
        update_period=1 / 30.0,
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


@configclass
class ObservationsCfg(SingleArmObservationsCfg):

    @configclass
    class PolicyCfg(SingleArmObservationsCfg.PolicyCfg):
        top = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False}
        )

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        pick_cube = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):

    success = DoneTerm(
        func=mdp.cube_height_above_base,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
            "height_threshold": 0.20,
        },
    )


@configclass
class LiftCubeEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the lift cube environment."""

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    task_description: str = "Lift the red cube up."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "cube",
                    pose_range={
                        "x": (-0.075, 0.075),
                        "y": (-0.075, 0.075),
                        "z": (0.0, 0.0),
                        "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180),
                    },
                ),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.005, 0.005),
                        "roll": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "pitch": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "yaw": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                    },
                    convention="opengl",
                ),
            ],
        )


@configclass
class LiftCubeDigitalTwinEnvCfg(LiftCubeEnvCfg, ManagerBasedRLDigitalTwinEnvCfg):
    """Configuration for the lift cube digital twin environment."""

    rgb_overlay_mode: str = "background"

    rgb_overlay_paths: dict[str, str] = {"front": "greenscreen/background-lift-cube.png"}

    render_objects: list[SceneEntityCfg] = [
        SceneEntityCfg("cube"),
        SceneEntityCfg("robot"),
    ]
