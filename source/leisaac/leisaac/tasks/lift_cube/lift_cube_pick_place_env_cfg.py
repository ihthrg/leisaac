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
TARGET_MARKER_SIZE = 0.09
TARGET_MARKER_HEIGHT = 0.004
TARGET_MARKER_COLOR = (1.0, 1.0, 1.0)
TARGET_RANDOM_RANGE = 0.06
"""Independent per-episode randomization half-range (m) applied to both the square target and
the circle target, in x and y."""


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
            func=mdp.cube_placed_on_correct_target,
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "target_cfg": SceneEntityCfg("target"),
                "circle_target_cfg": SceneEntityCfg("circle_target"),
                "robot_cfg": SceneEntityCfg("robot"),
                "center_x": 0.0,  # overwritten in LiftCubePickPlaceEnvCfg.__post_init__
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
            "circle_target_cfg": SceneEntityCfg("circle_target"),
            "robot_cfg": SceneEntityCfg("robot"),
            "center_x": 0.0,  # overwritten in LiftCubePickPlaceEnvCfg.__post_init__
        },
    )


@configclass
class LiftCubePickPlaceEnvCfg(LiftCubeEnvCfg):
    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(env_spacing=8.0)
    observations: LiftCubePickPlaceObservationsCfg = LiftCubePickPlaceObservationsCfg()
    terminations: LiftCubePickPlaceTerminationsCfg = LiftCubePickPlaceTerminationsCfg()

    task_description: str = (
        "Pick up the red cube, hold it briefly above the table center, then place it on the white circular"
        " target if it was picked up from the right side or the white square target if picked up from the"
        " left side, and return the arm to rest."
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        cube_pos = self.scene.cube.init_state.pos
        center_x = cube_pos[0]
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
                    diffuse_color=TARGET_MARKER_COLOR,
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
                    diffuse_color=TARGET_MARKER_COLOR,
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=1.0,
                    metallic=0.0,
                    opacity=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=circle_target_pos),
        )

        # Which target matches the cube's original spawn side (right of center_x -> circle,
        # left -> square) is resolved at runtime by mdp.cube_placed_on_correct_target/cube_pick_place_done.
        self.terminations.success.params["center_x"] = center_x
        self.observations.subtask_terms.place_cube.params["center_x"] = center_x

        setattr(
            self.events,
            "domain_randomize_target",
            randomize_object_uniform(
                "target",
                pose_range={
                    "x": (-TARGET_RANDOM_RANGE, TARGET_RANDOM_RANGE),
                    "y": (-TARGET_RANDOM_RANGE, TARGET_RANDOM_RANGE),
                    "z": (0.0, 0.0),
                },
            ),
        )
        setattr(
            self.events,
            "domain_randomize_circle_target",
            randomize_object_uniform(
                "circle_target",
                pose_range={
                    "x": (-TARGET_RANDOM_RANGE, TARGET_RANDOM_RANGE),
                    "y": (-TARGET_RANDOM_RANGE, TARGET_RANDOM_RANGE),
                    "z": (0.0, 0.0),
                },
            ),
        )
