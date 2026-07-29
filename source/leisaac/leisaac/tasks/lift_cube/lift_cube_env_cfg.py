"""
python scripts/datagen/state_machine/generate.py --task LeIsaac-SO101-LiftCubePickPlace-v0 \
    --num_envs 1 --device cuda --enable_cameras --num_demos 3 \
    --record --use_lerobot_recorder --lerobot_dataset_fps 30 --step_hz 60
"""

import isaaclab.sim as sim_utils
import leisaac.enhance.envs.mdp as enhance_mdp
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
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
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.env_utils import delete_attribute
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

# U20CAM-1080P-S1 の仕様に合わせた魚眼カメラ設定
# Note: IsaacLab 2.3 uses `FisheyeCameraCfg` with `projection_type` and the
# `fisheye_*` width/height/centre parameters instead of the older
# `fisheye_nominal_focal_length` / `fisheye_model` API.
u20cam_spawn_cfg = sim_utils.FisheyeCameraCfg(
    projection_type="fisheyePolynomial",
    focal_length=2.8,  # 実機公称値
    focus_distance=400.0,
    horizontal_aperture=4.8,  # 1/3"センサー幅
    vertical_aperture=3.6,  # 1/3"センサー高さ（4:3, 640x480に整合）
    clipping_range=(0.01, 50.0),
    lock_camera=True,
    fisheye_nominal_width=U20CAM_WIDTH,
    fisheye_nominal_height=U20CAM_HEIGHT,
    fisheye_optical_centre_x=U20CAM_WIDTH / 2.0,
    fisheye_optical_centre_y=U20CAM_HEIGHT / 2.0,
    fisheye_max_fov=130.0,  # 実機公称の対角FOV（現状120.0も要修正）
)

ARM_SERVO_NAME_PATTERNS = ("sts3215", "servo", "motor")
"""Case-insensitive substrings that pick out the arm's servo meshes, matched against each mesh's
prim path below the robot root.

These are a guess, and are meant to be corrected rather than trusted: the SO-101 asset is a binary
USD whose mesh names cannot be read without Isaac Sim. ``paint_meshes_by_name`` prints both the
paths it matched and the ones it did not, so one run says whether these are right and shows exactly
what to use instead if they are not."""

ARM_SERVO_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.05, 0.05, 0.05),
    emissive_color=(0.0, 0.0, 0.0),
    roughness=0.4,
    metallic=0.0,
    opacity=1.0,
)
"""Near-black finish for the servos.

Not pure black, which clips against shadow, and glossier than the body below because a fully rough
black surface reflects almost nothing and renders as a flat silhouette; the sheen is what keeps the
servo bodies readable in the recorded frames."""

ARM_BODY_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.9, 0.9, 0.9),
    emissive_color=(0.0, 0.0, 0.0),
    roughness=0.7,
    metallic=0.0,
    opacity=1.0,
)
"""Near-white finish for everything else.

Held off pure white because the pick-and-place variant's markers are exactly ``(1, 1, 1)``
(``TARGET_MARKER_COLOR``) and the arm would be hard to separate from one it is passing over. Matter
than the servos for the opposite reason to theirs: on a bright surface it is the highlights that
saturate and flatten the links."""


@configclass
class LiftCubeSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the lift cube task."""

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.058, -0.021), rot=(0.0, 0.0, 0.9812086, 0.1929501), convention="ros"
        ),
        data_types=["rgb"],
        spawn=u20cam_spawn_cfg,
        width=U20CAM_WIDTH,
        height=U20CAM_HEIGHT,
        update_period=1 / 30.0,
    )

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.02, -0.21, 0.45), rot=(0.0, 0.0, 1.0, 0.0), convention="ros"),
        data_types=["rgb"],
        spawn=u20cam_spawn_cfg,
        width=U20CAM_WIDTH,
        height=U20CAM_HEIGHT,
        update_period=1 / 30.0,
    )
    light = AssetBaseCfg(
        # Outside the per-environment namespace on purpose. A dome light illuminates from an
        # infinite sphere, so it is not confined to the environment it is parented under: one per
        # environment would stack, and `--num_envs 4` would render every camera at four times the
        # intensity of the single-environment dataset. At `--num_envs 1` this is identical.
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    def __post_init__(self):
        super().__post_init__()
        # `front` is inherited from SingleArmTaskSceneCfg but this task now uses
        # wrist + top instead, so drop it from the scene (and thus from recording).
        delete_attribute(self, "front")


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

    def __post_init__(self):
        super().__post_init__()
        # drop the inherited `front` observation term to match the scene above
        delete_attribute(self.policy, "front")


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

        # A white arm with black servos needs a material on each mesh, not one on the robot's root
        # prim: `UsdFileCfg.visual_material` binds `strongerThanDescendants`, which would override
        # any per-mesh binding underneath it. Done as a startup event because the meshes only exist
        # once the asset has been spawned.
        setattr(
            self.events,
            "paint_robot",
            EventTerm(
                func=enhance_mdp.paint_meshes_by_name,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "match_patterns": ARM_SERVO_NAME_PATTERNS,
                    "match_material": ARM_SERVO_MATERIAL,
                    "default_material": ARM_BODY_MATERIAL,
                },
            ),
        )

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
                # `front` is not part of this scene (see LiftCubeSceneCfg.__post_init__), so only
                # the overhead camera is perturbed. The ranges stand in for the mounting tolerance
                # of the real rig; the convention matches the camera's offset above.
                randomize_camera_uniform(
                    "top",
                    pose_range={
                        "x": (-0.01, 0.01),
                        "y": (-0.01, 0.01),
                        "z": (-0.01, 0.01),
                        "roll": (-1.0 * torch.pi / 180, 1.0 * torch.pi / 180),
                        "pitch": (-1.0 * torch.pi / 180, 1.0 * torch.pi / 180),
                        "yaw": (-1.0 * torch.pi / 180, 1.0 * torch.pi / 180),
                    },
                    convention="ros",
                ),
            ],
        )


@configclass
class LiftCubeDigitalTwinEnvCfg(LiftCubeEnvCfg, ManagerBasedRLDigitalTwinEnvCfg):
    """Configuration for the lift cube digital twin environment."""

    rgb_overlay_mode: str = "background"

    # avoid using the `front` camera overlay so front camera won't be used
    # in digital-twin recording; use the wrist camera instead
    rgb_overlay_paths: dict[str, str] = {"wrist": "greenscreen/background-lift-cube.png"}

    render_objects: list[SceneEntityCfg] = [
        SceneEntityCfg("cube"),
        SceneEntityCfg("robot"),
    ]
