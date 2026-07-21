import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.utils.domain_randomization import randomize_object_uniform

from ..template import SingleArmTerminationsCfg
from . import mdp
from .lift_cube_env_cfg import LiftCubeEnvCfg, LiftCubeSceneCfg, ObservationsCfg

TARGET_OFFSET_X = 0.18
TARGET_MARKER_Z_OFFSET = -0.018
TARGET_MARKER_SIZE = 0.10
TARGET_MARKER_HEIGHT = 0.004


@configclass
class LiftCubePickPlaceObservationsCfg(ObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        pick_cube = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )
        place_cube = ObsTerm(
            func=mdp.cube_placed_on_target,
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "target_cfg": SceneEntityCfg("target"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class LiftCubePickPlaceTerminationsCfg(SingleArmTerminationsCfg):
    success = DoneTerm(
        func=mdp.cube_pick_place_done,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("target"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class LiftCubePickPlaceEnvCfg(LiftCubeEnvCfg):
    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(env_spacing=8.0)
    observations: LiftCubePickPlaceObservationsCfg = LiftCubePickPlaceObservationsCfg()
    terminations: LiftCubePickPlaceTerminationsCfg = LiftCubePickPlaceTerminationsCfg()

    task_description: str = "Pick up the red cube, place it on the green target, and return the arm to rest."

    def __post_init__(self) -> None:
        super().__post_init__()

        cube_pos = self.scene.cube.init_state.pos
        target_pos = (
            cube_pos[0] + TARGET_OFFSET_X,
            cube_pos[1],
            cube_pos[2] + TARGET_MARKER_Z_OFFSET,
        )
        circle_target_pos = (
            cube_pos[0] - TARGET_OFFSET_X,
            cube_pos[1],
            cube_pos[2] + TARGET_MARKER_Z_OFFSET,
        )
        self.scene.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=sim_utils.CuboidCfg(
                size=(TARGET_MARKER_SIZE, TARGET_MARKER_SIZE, TARGET_MARKER_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.8, 0.1),
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=1.0,
                    metallic=0.0,
                    opacity=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=target_pos),
        )
        self.scene.circle_target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CircleTarget",
            spawn=sim_utils.CylinderCfg(
                radius=TARGET_MARKER_SIZE / 2,
                height=TARGET_MARKER_HEIGHT,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.65, 0.9),
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=1.0,
                    metallic=0.0,
                    opacity=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=circle_target_pos),
        )

        setattr(
            self.events,
            "domain_randomize_target",
            randomize_object_uniform(
                "target",
                pose_range={
                    "x": (-0.03, 0.03),
                    "y": (-0.03, 0.03),
                    "z": (0.0, 0.0),
                },
            ),
        )
