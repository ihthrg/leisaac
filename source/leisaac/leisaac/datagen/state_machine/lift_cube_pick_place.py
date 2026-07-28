"""State machine for the conditional lift-cube pick-and-place (sort) task."""

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS
from leisaac.tasks.lift_cube.mdp import cube_placed_on_correct_target

from .base import StateMachineBase
from .planar_arm_model import calibrate_planar_arm, wrap_to_pi

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_ARM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

_PROBE_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
"""Joints the kinematic calibration solves for, in the order ``calibrate_planar_arm`` expects."""

_GRIPPER_OPEN_RAD = 1.0
_GRIPPER_CLOSE_RAD = 0.01
"""Gripper joint angles (rad) the kinematics are calibrated at: the two ends of its travel.

``_GRIPPER_CLOSE_RAD`` is what the jaws are ramped towards when gripping, but they are not expected
to reach it: the cube stops them first, and the grip holds just inside wherever that turns out to
be -- see ``_GRIP_CONTACT_LAG_RAD`` below."""

_CUBE_SIZE = 0.03
"""Edge length (m) of the cube this task picks up."""

_GRIP_CONTACT_LAG_RAD = 0.12
"""How far the gripper joint may fall behind its command before the cube is deemed to be in the way.

The jaws are closed by ramping the command shut and watching the joint follow it. Free, the joint
trails the ramp by roughly ``rate * damping / stiffness``, which the ramp rate is chosen to keep
near 0.07 rad; once the cube blocks it the gap grows without bound. This sits between the two.

Closing to a *measured* width was tried first and does not work. Isaac Sim only reports the moving
finger, so a commanded width is only as good as the assumption that the stationary finger's face
sits exactly where the moving one rests when shut. In sim it does not: aiming for a 28 mm grip on
the 30 mm cube left the jaws wide enough that the cube was merely nudged 5 mm across and never
pinched. Closing onto the object needs no such assumption, and adapts to whatever it finds."""

_GRIP_SQUEEZE_RAD = 0.05
"""Interference held past the angle the cube stopped the jaws at.

The actuator is position controlled, so this is what sets the grip force: about
``stiffness * this`` = 0.9 N m, which at the finger's ~7 cm radius is over ten newtons of pinch --
ample for a light cube, and far short of commanding the jaws fully shut, which would try to drive
them 0.5 rad into it."""

_GRIP_SETTLE_STEPS = 30
"""Steps (0.5s) spent easing the command back from where contact was noticed to the squeeze."""

_GRASP_FACE_CLEARANCE = 0.005
"""Gap (m) left between the stationary finger and the cube face while descending.

The tracked point sits ``_CUBE_SIZE / 2 + this`` from the stationary finger, rather than in the
middle of the jaws. Both keep the cube off the stationary finger on the way down, but the middle
is needlessly far: the moving finger then has to travel the whole way in before it touches, and
shoves the cube ahead of it until it pins. Measured in sim at the midpoint, that shove was 18 mm;
from here it is 5 mm."""

_GRIPPER_REST_RAD = math.radians(-10.0)
"""Gripper angle at rest -- closed, matching ``_REST_POSE_DEG``.

Episodes start and end with the jaws shut, so a recorded episode begins and ends in the same
physical state the arm idles in."""

_WRIST_ROLL_DEG = 90.0
"""``wrist_roll`` is held here for the whole episode.

With the gripper pointing straight down this joint only spins the fingers about the vertical
approach axis, so the cube can be grasped at any value; keeping it constant means the joint never
has to unwind mid-episode, and it matches the requested rest and hold postures."""

_REST_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 90.0,
    "wrist_flex": 50.0,
    "wrist_roll": _WRIST_ROLL_DEG,
    "gripper": -10.0,
}

_HOLD_LINK_PITCHES_DEG = (90.0, 0.0, 0.0)
"""In-air hold posture ("middle position"), given as the absolute pitch of each arm link.

This states the requested shape directly: the segment leaving joint ID2 (the upper arm) is
**vertical** (+90 deg) and everything from joint ID3 outwards -- the forearm *and* the gripper --
is **horizontal** (0 deg). The joint angles are solved from these three pitches at start-up by
``PlanarArmModel.joints_for_link_pitches``, against kinematics measured on the actual robot.

Stating the posture as link pitches rather than joint angles is what makes it correct. Every pitch
joint carries through to the links beyond it, so the same ``elbow_flex`` value means a different
forearm pitch at every ``shoulder_lift``; and ``wrist_flex = 0`` leaves the gripper parallel to the
forearm only if those two links happen to share a zero offset, which on this arm they do not.
Earlier revisions fixed joint angles by hand and produced an arm that looked stretched out because
the links did not end up at the intended pitches."""

_CALIB_DELTA_DEG = 45.0
"""Probe displacement used to measure the kinematics. Large enough for well-conditioned circle
fits, small enough to stay inside every joint limit from the all-zero probe pose."""

_CALIB_RESIDUAL_WARN = 0.005
"""Model-vs-measured disagreement (m) at the hold pose above which ``setup()`` warns loudly.

The calibration is exact geometry, so anything beyond a millimetre or two means the probe poses
were not measured cleanly and every jaw target inherits the error."""

_JOINT_DAMPING = 3.0
"""Joint damping override (the asset default is 0.60, tuned for teleoperation).

Damping sets how fast the implicit PD actuator closes on its commanded angle: the response time is
roughly ``damping / stiffness``, so the 10.0 used previously meant 0.56 s, more than a third of
some phases. At 3.0 that drops to 0.17 s -- still comfortably overdamped, so the arm does not
overshoot and shake the cube loose, but fast enough that every phase settles well before it
ends."""

_SUCCESS_GRIPPER_THRESHOLD = -1.0
"""``grasp_threshold`` handed to the placement check at the end of an episode.

That check normally also insists the gripper is open, as evidence the cube was let go. Episodes
now finish with the jaws shut at the rest pose, so that test would reject every success. Any value
below the rest angle disables it, and nothing is really lost: the check still requires the cube to
be stationary on the marker, which cannot happen if it were still being carried."""

_REST_POSE_TOLERANCE_DEG = 30.0
"""Per-joint +/- tolerance (deg) used by ``_is_at_rest``. Matches the shared
``SO101_FOLLOWER_REST_POSE_RANGE`` for every joint except ``wrist_roll`` -- this task rests at
90 deg rather than 0 deg, so it uses its own local check instead of the shared one, to avoid
loosening the tolerance for other tasks."""

_HOLD_HEIGHT_ABOVE_TABLE = 0.20
"""Jaw height (m) above the cube/target while carrying."""

_HOVER_CLEARANCE = 0.07
"""Jaw height (m) above the cube when hovering, before lowering to grasp.

Kept modest on purpose: a strictly vertical gripper is hardest to reach high up and far out, so a
tall hover point is where the solver has to start tilting the gripper. 7 cm is ample clearance to
come down onto a cube."""

_RELEASE_CLEARANCE = 0.03
"""Jaw height (m) above the target marker when releasing. The jaw holds the cube at its centre and
the markers sit slightly below the cube's resting centre height, so this drops the cube from just
above the marker."""

_GRASP_DEPTH_BELOW_CENTER = 0.005
"""How far (m) below the cube's centre the jaw is aimed, so the fingers straddle the cube rather
than grazing its top face."""

# Phase boundaries (state-machine step count). Each `env.step()` advances physics by one sim tick,
# so a 60Hz simulation (`--step_hz 60`) needs `round(seconds * 60)` steps for the intended
# duration. `LeRobotRecorderManager` separately downsamples those steps to the requested dataset
# rate (`--lerobot_dataset_fps 30`, i.e. every second step here).
_APPROACH_STEPS = 180  # phase ends at 180 (3.0s): rest -> hovering above the cube
_LOWER_TO_CUBE_END = 300  # +120 (2.0s): descend onto the cube
_GRASP_CLOSE_STEPS = 150  # 2.5s ramping the jaws shut, slow enough that the contact lag stands out
_GRASP_END = 510  # +210 (3.5s): close onto the cube, then hold still for 1.0s so the grip settles
_LIFT_CUBE_END = 570  # +60 (1.0s)
_MOVE_TO_MIDDLE_END = 660  # +90 (1.5s)
_HOLD_MIDDLE_END = 840  # +180 (3.0s) <- user requirement: hold ~3s at the "middle position"
_MOVE_ABOVE_TARGET_END = 930  # +90 (1.5s)
_LOWER_TO_TARGET_END = 990  # +60 (1.0s)
_RELEASE_END = 1050  # +60 (1.0s)
_LIFT_GRIPPER_END = 1110  # +60 (1.0s)
_TOTAL_STEPS = 1410  # +300 (5.0s): return home, ends the episode

# DEBUG: (step, phase-name) pairs used by the diagnostic print in get_action() -- remove once the
# grasp and hold posture are confirmed correct in sim.
_PHASE_START_NAMES = (
    (0, "approach_hover"),
    (_APPROACH_STEPS, "lower_to_cube"),
    (_LOWER_TO_CUBE_END, "grasp"),
    (_GRASP_END, "lift_cube"),
    (_LIFT_CUBE_END, "move_to_middle"),
    (_MOVE_TO_MIDDLE_END, "hold_middle"),
    (_HOLD_MIDDLE_END, "move_above_target"),
    (_MOVE_ABOVE_TARGET_END, "lower_to_target"),
    (_LOWER_TO_TARGET_END, "release"),
    (_RELEASE_END, "lift_gripper"),
    (_LIFT_GRIPPER_END, "return_home"),
)


def _ease(alpha: float) -> float:
    """Smoothstep easing, so each phase starts and ends at zero joint velocity.

    Plain linear interpolation would step the commanded velocity discontinuously at every phase
    boundary, which jolts a carried cube.
    """
    alpha = min(max(alpha, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


def _is_at_rest(joint_pos: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    """Check whether every joint is within ``_REST_POSE_TOLERANCE_DEG`` of ``_REST_POSE_DEG``."""
    is_rest = torch.ones(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)
    joint_pos_deg = joint_pos / torch.pi * 180.0
    for joint_name, target_deg in _REST_POSE_DEG.items():
        joint_idx = joint_names.index(joint_name)
        is_rest = torch.logical_and(
            is_rest,
            torch.logical_and(
                joint_pos_deg[:, joint_idx] > target_deg - _REST_POSE_TOLERANCE_DEG,
                joint_pos_deg[:, joint_idx] < target_deg + _REST_POSE_TOLERANCE_DEG,
            ),
        )
    return is_rest


class LiftCubePickPlaceStateMachine(StateMachineBase):
    """State machine for the conditional lift-cube pick-and-place (sort) task.

    Picks up the cube, holds it for ~3 seconds at the requested "middle position" (see
    ``_HOLD_LINK_PITCHES_DEG``), then places it on whichever target matches its original spawn
    side: the circle target if the cube spawned to the right (+X) of the nominal centre, or the
    square target if it spawned to the left. Finally returns the arm to the SO-101 rest pose.

    The whole episode is planned in **joint space**, and this task's action term is configured as a
    joint-position action to match (see ``devices/action_process.py``). That is deliberate. The
    SO-101 has five arm joints, so driving it through a differential-IK action that consumes a
    6-DoF pose is over-constrained: the Jacobian is 6x5 and always has a left-null direction, and
    the solver comes to rest wherever the residual pose error lies entirely in that direction
    rather than at the commanded pose. Measured in sim, that trap left the jaw 14 cm from its
    target at the end of a 3 s approach, so the fingers closed on air, and let the arm drift 23 cm
    away from a *static* hold command, so the hold posture never looked like the requested one.
    Both are rank deficiencies, not tuning problems, and no gain or timing change removes them.

    Planning in joint space removes the ambiguity: the four unknowns this task actually needs
    (``shoulder_pan`` for the heading, plus the three pitch joints for a top-down reach in the
    arm's own plane) are exactly determined, and are solved in closed form by
    ``planar_arm_model.PlanarArmModel``, whose parameters are measured from the robot in
    ``setup()``. Easing between solved configurations also keeps the arm away from the stretched,
    near-singular poses the IK solver kept falling into.

    Each episode starts by snapping the arm to the rest pose (see ``pre_step()``) before any action
    is recorded, so every recorded episode begins from a consistent rest state.
    """

    TOTAL_STEPS: int = _TOTAL_STEPS

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._model = None
        self._arm_names: list[str] = []
        self._arm_indices: torch.Tensor | None = None
        self._joint_limits: torch.Tensor | None = None
        self._rest_joint_pos: torch.Tensor | None = None
        self._rest_arm: torch.Tensor | None = None
        self._hold_arm: torch.Tensor | None = None
        self._gripper_index: int = -1
        self._grip_hold: float | None = None
        self._grip_contact: tuple[int, float] = (0, _GRIPPER_CLOSE_RAD)
        self._start_arm: torch.Tensor | None = None
        self._home_start_arm: torch.Tensor | None = None
        self._cube_pos_w: torch.Tensor | None = None
        self._center_x: float = 0.0
        self._target_is_circle: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        """Measure the arm's kinematics, then solve the fixed poses the episode is built from.

        Ten forward-kinematics probes pin down every parameter in closed form, because sweeping
        one joint sends the jaw around a circle centred on that joint (see
        ``planar_arm_model.calibrate_planar_arm``). Measuring beats hard-coding here: the link
        lengths and joint-angle offsets that come out of this are exactly the quantities earlier
        revisions guessed at, and guessed wrong.

        Two details of the probe pose matter and are easy to get wrong:

        * ``wrist_roll`` moves the jaw frame, so it is held at the value used during the episode.
        * Only one of the SO-101's fingers moves, so the jaw frame Isaac Sim reports swings a long
          way as the gripper opens. The tenth probe repeats the reference pose with the jaws open
          so the model can track the midpoint between the fingertips; aiming the closed position
          at a cube instead puts the *stationary* finger where the cube is and shoves it aside.

        The caller resets the environment after ``setup()``, so none of these probe poses leak
        into the first recorded episode.

        ``center_x`` (the left/right decision threshold) is derived from the cube's *nominal*
        (pre-randomisation) spawn point, i.e. ``env.cfg.scene.cube.init_state.pos`` -- the same
        fixed value the env config uses to place the two targets and to seed ``center_x`` in
        ``mdp.cube_placed_on_correct_target``/``cube_pick_place_done``. Reading it from the config
        rather than from the live, possibly-already-randomised ``root_pos_w`` guarantees the state
        machine's target choice always agrees with the success check's.
        """
        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        joint_names = list(robot.data.joint_names)

        # The action manager lays the arm term out in the articulation's own joint order, so build
        # every command vector in that order rather than in ``_ARM_JOINT_NAMES`` order.
        self._arm_names = [name for name in joint_names if name in _ARM_JOINT_NAMES]
        self._arm_indices = torch.tensor([joint_names.index(name) for name in self._arm_names], device=env.device)
        self._joint_limits = torch.tensor(
            [[math.radians(value) for value in SO101_FOLLOWER_USD_JOINT_LIMLITS[name]] for name in self._arm_names],
            device=env.device,
        )

        def write_pose(pose_deg: dict[str, float]) -> torch.Tensor:
            joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
            for idx, name in enumerate(joint_names):
                joint_pos[:, idx] = math.radians(pose_deg.get(name, 0.0))
            robot.write_joint_state_to_sim(position=joint_pos, velocity=torch.zeros_like(joint_pos))
            # Writing the joint *state* alone is not enough. The actuator keeps driving towards
            # whatever position target was last written -- all zeros at start-up -- so the physics
            # step below immediately drags the arm back out of the pose that was just written. In
            # sim that cost about 9 deg of a 45 deg probe, which showed up as a joint gain of
            # 0.806 instead of 1.0 and correctly aborted the calibration. Commanding the target
            # too makes the step a no-op and the pose actually holds.
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            return joint_pos

        probe_indices = [joint_names.index(name) for name in _PROBE_JOINT_NAMES]

        def probe(
            pan_deg: float,
            lift_deg: float,
            elbow_deg: float,
            wrist_flex_deg: float,
            wrist_roll_deg: float,
            gripper_deg: float,
        ):
            write_pose({
                "shoulder_pan": pan_deg,
                "shoulder_lift": lift_deg,
                "elbow_flex": elbow_deg,
                "wrist_flex": wrist_flex_deg,
                "wrist_roll": wrist_roll_deg,
                "gripper": gripper_deg,
            })
            jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :]
            jaw_pos_b = quat_apply(quat_inv(robot.data.root_quat_w), jaw_pos_w - robot.data.root_pos_w)
            # Report the angles the arm actually settled at, not the ones asked for, so the
            # calibration stays exact even if a pose is held imperfectly.
            achieved = robot.data.joint_pos[0, probe_indices]
            return (
                tuple(float(value) for value in jaw_pos_b[0]),
                tuple(float(value) for value in achieved),
            )

        self._model = calibrate_planar_arm(
            probe,
            delta_deg=_CALIB_DELTA_DEG,
            wrist_roll_deg=_WRIST_ROLL_DEG,
            gripper_closed_deg=math.degrees(_GRIPPER_CLOSE_RAD),
            gripper_open_deg=math.degrees(_GRIPPER_OPEN_RAD),
            grasp_offset=0.5 * _CUBE_SIZE + _GRASP_FACE_CLEARANCE,
        )
        self._gripper_index = joint_names.index("gripper")

        hold_lift, hold_elbow, hold_wrist_flex = self._model.joints_for_link_pitches(*_HOLD_LINK_PITCHES_DEG)
        self._hold_arm = self._arm_command(
            env,
            {
                "shoulder_pan": 0.0,
                "shoulder_lift": hold_lift,
                "elbow_flex": hold_elbow,
                "wrist_flex": hold_wrist_flex,
                "wrist_roll": math.radians(_WRIST_ROLL_DEG),
            },
        )
        self._rest_arm = self._arm_command(env, {name: math.radians(_REST_POSE_DEG[name]) for name in self._arm_names})

        self._center_x = float(env.cfg.scene.cube.init_state.pos[0])

        # DEBUG: report the measured model and check that the hold posture actually lands where the
        # model says it should -- remove together with the per-phase print once confirmed in sim.
        hold_deg = (0.0, math.degrees(hold_lift), math.degrees(hold_elbow), math.degrees(hold_wrist_flex))
        closed_tip = probe(*hold_deg, _WRIST_ROLL_DEG, math.degrees(_GRIPPER_CLOSE_RAD))[0]
        open_tip = probe(*hold_deg, _WRIST_ROLL_DEG, math.degrees(_GRIPPER_OPEN_RAD))[0]
        fraction = self._model.grasp_fraction
        measured = tuple(a + fraction * (b - a) for a, b in zip(closed_tip, open_tip))
        predicted = self._model.jaw_in_plane(hold_lift, hold_elbow, hold_wrist_flex)
        residual = math.dist(self._model.to_plane(measured), predicted)
        self._rest_joint_pos = write_pose(_REST_POSE_DEG)
        print(
            f"[LiftCubePickPlace][setup] {self._model.describe()}\n"
            f"[LiftCubePickPlace][setup] hold joints (deg): shoulder_lift={math.degrees(hold_lift):.2f} "
            f"elbow_flex={math.degrees(hold_elbow):.2f} wrist_flex={math.degrees(hold_wrist_flex):.2f} "
            f"-> link pitches {_HOLD_LINK_PITCHES_DEG} deg, jaw at forward={predicted[0]:.4f} "
            f"height={predicted[1]:.4f} m, model-vs-measured residual={residual * 1000.0:.1f} mm, "
            f"center_x={self._center_x}"
        )
        if residual > _CALIB_RESIDUAL_WARN:
            print(
                "[LiftCubePickPlace][setup] WARNING: the measured kinematics disagree with the robot by "
                f"{residual * 1000.0:.1f} mm at the hold pose. Every jaw target will be off by roughly that "
                "much; check that the calibration probes are being held steady."
            )

    def check_success(self, env) -> bool:
        """Return True if the cube is on its correct target and the arm is at rest."""
        placed = cube_placed_on_correct_target(
            env,
            cube_cfg=SceneEntityCfg("cube"),
            target_cfg=SceneEntityCfg("target"),
            circle_target_cfg=SceneEntityCfg("circle_target"),
            robot_cfg=SceneEntityCfg("robot"),
            center_x=self._center_x,
            grasp_threshold=_SUCCESS_GRIPPER_THRESHOLD,
        )

        robot = env.scene["robot"]
        if self._rest_joint_pos is not None:
            robot.write_joint_state_to_sim(
                position=self._rest_joint_pos,
                velocity=torch.zeros_like(self._rest_joint_pos),
            )
            env.scene.update(dt=env.physics_dt)
        at_rest = _is_at_rest(robot.data.joint_pos, robot.data.joint_names)

        return bool(torch.logical_and(placed, at_rest).all().item())

    def pre_step(self, env) -> None:
        """Snap to the rest pose before the first step of an episode.

        Every recorded episode then starts from the same state. Nothing is written after that --
        the return-home phase commands the rest joint angles like any other phase, so the arm
        drives there under normal actuation instead of being teleported while recording runs.
        """
        if self._step_count == 0 and self._rest_joint_pos is not None:
            robot = env.scene["robot"]
            robot.write_joint_state_to_sim(
                position=self._rest_joint_pos,
                velocity=torch.zeros_like(self._rest_joint_pos),
            )
            env.scene.update(dt=env.physics_dt)

    def get_action(self, env) -> torch.Tensor:
        """Compute the action for the current step: five arm joint angles plus the gripper.

        Cube and target positions describe where the *grasp centre* should be -- the point midway
        between the two fingertips with the jaws open, which is what the kinematic model tracks.
        Only one finger moves on this arm, so that is a very different place from the jaw frame
        Isaac Sim reports.

        Phases either hold a solved configuration or ease between two of them, so consecutive
        phases always agree at their shared boundary and the commanded joint velocity is zero
        there.
        """
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=_JOINT_DAMPING)

        step = self._step_count

        if step == 0:
            # The cube only moves once the gripper touches it, so freezing its pose here keeps the
            # whole approach aimed at one fixed point instead of chasing a cube it just nudged.
            self._cube_pos_w = env.scene["cube"].data.root_pos_w.clone()
            self._target_is_circle = self._cube_pos_w[:, 0] > self._center_x
            self._start_arm = robot.data.joint_pos[:, self._arm_indices].clone()

        place_pos_w = torch.where(
            self._target_is_circle.unsqueeze(-1),
            env.scene["circle_target"].data.root_pos_w,
            env.scene["target"].data.root_pos_w,
        )

        def raised(base_pos: torch.Tensor, height: float) -> torch.Tensor:
            shifted = base_pos.clone()
            shifted[:, 2] += height
            return shifted

        def solve(jaw_target_w: torch.Tensor) -> torch.Tensor:
            return self._solve_arm(robot, jaw_target_w)

        hover = raised(self._cube_pos_w, _HOVER_CLEARANCE)
        grasp = raised(self._cube_pos_w, -_GRASP_DEPTH_BELOW_CENTER)
        lifted = raised(self._cube_pos_w, _HOLD_HEIGHT_ABOVE_TABLE)
        place_hover = raised(place_pos_w, _HOLD_HEIGHT_ABOVE_TABLE)
        place_release = raised(place_pos_w, _RELEASE_CLEARANCE)

        if step < _APPROACH_STEPS:
            progress = step / _APPROACH_STEPS
            arm = self._blend(self._start_arm, solve(hover), progress)
            # Opens on the way over rather than starting open, so the episode begins with the arm
            # in the state it idles in: jaws shut.
            gripper = self._ease_scalar(_GRIPPER_REST_RAD, _GRIPPER_OPEN_RAD, progress)
        elif step < _LOWER_TO_CUBE_END:
            arm = self._blend(solve(hover), solve(grasp), self._progress(step, _APPROACH_STEPS, _LOWER_TO_CUBE_END))
            gripper = _GRIPPER_OPEN_RAD
        elif step < _GRASP_END:
            arm = solve(grasp)
            gripper = self._closing_gripper(robot, step)
        elif step < _LIFT_CUBE_END:
            arm = self._blend(solve(grasp), solve(lifted), self._progress(step, _GRASP_END, _LIFT_CUBE_END))
            gripper = self._grip_command
        elif step < _MOVE_TO_MIDDLE_END:
            arm = self._blend(solve(lifted), self._hold_arm, self._progress(step, _LIFT_CUBE_END, _MOVE_TO_MIDDLE_END))
            gripper = self._grip_command
        elif step < _HOLD_MIDDLE_END:
            arm = self._hold_arm.clone()
            gripper = self._grip_command
        elif step < _MOVE_ABOVE_TARGET_END:
            arm = self._blend(
                self._hold_arm, solve(place_hover), self._progress(step, _HOLD_MIDDLE_END, _MOVE_ABOVE_TARGET_END)
            )
            gripper = self._grip_command
        elif step < _LOWER_TO_TARGET_END:
            arm = self._blend(
                solve(place_hover),
                solve(place_release),
                self._progress(step, _MOVE_ABOVE_TARGET_END, _LOWER_TO_TARGET_END),
            )
            gripper = self._grip_command
        elif step < _RELEASE_END:
            arm = solve(place_release)
            # Eased open over the first half of the phase rather than sprung: snapping the jaws
            # apart around a cube resting on the marker can flick it back off.
            gripper = self._ease_scalar(
                self._grip_command, _GRIPPER_OPEN_RAD, 2.0 * self._progress(step, _LOWER_TO_TARGET_END, _RELEASE_END)
            )
        elif step < _LIFT_GRIPPER_END:
            arm = self._blend(
                solve(place_release), solve(place_hover), self._progress(step, _RELEASE_END, _LIFT_GRIPPER_END)
            )
            gripper = _GRIPPER_OPEN_RAD
        else:
            if self._home_start_arm is None:
                self._home_start_arm = solve(place_hover)
            progress = self._progress(step, _LIFT_GRIPPER_END, _TOTAL_STEPS)
            arm = self._blend(self._home_start_arm, self._rest_arm, progress)
            # Shuts on the way home, so the arm ends in its idle state. The success check is told
            # not to read the gripper because of it -- see ``_SUCCESS_GRIPPER_THRESHOLD``.
            gripper = self._ease_scalar(_GRIPPER_OPEN_RAD, _GRIPPER_REST_RAD, progress)

        self._log_phase_start(env, robot, step, arm)

        gripper_cmd = torch.full((env.num_envs, 1), gripper, device=env.device)
        return torch.cat([arm, gripper_cmd], dim=-1)

    def advance(self) -> None:
        """Advance the step counter; the episode ends once the return-home phase completes."""
        self._step_count += 1
        if self._step_count >= _TOTAL_STEPS:
            self._episode_done = True

    def reset(self) -> None:
        """Reset the state machine to its initial state for a new episode."""
        self._step_count = 0
        self._episode_done = False
        self._start_arm = None
        self._home_start_arm = None
        self._cube_pos_w = None
        self._target_is_circle = None
        self._grip_hold = None
        self._grip_contact = (0, _GRIPPER_CLOSE_RAD)

    # ------------------------------------------------------------------

    @property
    def _grip_command(self) -> float:
        """Gripper command to hold while the cube is being carried.

        Falls back to the fully-shut command when nothing was ever felt, which leaves the jaws
        closed on empty air -- the honest outcome for an episode that missed the cube, and one the
        success check will reject on its own.
        """
        return _GRIPPER_CLOSE_RAD if self._grip_hold is None else self._grip_hold

    def _closing_gripper(self, robot, step: int) -> float:
        """Ramp the jaws shut, stop at whatever they land on, and hold a fixed squeeze on it.

        The ramp is linear rather than eased so its rate -- and hence how far the joint lags behind
        it while it is still free to move -- stays constant and predictable; a smooth start is
        worth nothing here since the jaws are closing on empty air for the first half of it.

        Contact is only noticed once the command has already been driven ``_GRIP_CONTACT_LAG_RAD``
        past what the joint could reach, so the command is walked back to the squeeze over
        ``_GRIP_SETTLE_STEPS`` rather than dropped there in one step, which would jolt the cube
        just as it is being taken hold of.

        The command is one scalar for every environment, so contact is taken from whichever jaws
        stall first. With only the cube's pose randomised they all block within a hair of each
        other, so that is also the angle the rest want.
        """
        if self._grip_hold is None:
            progress = min((step - _LOWER_TO_CUBE_END) / _GRASP_CLOSE_STEPS, 1.0)
            command = _GRIPPER_OPEN_RAD + (_GRIPPER_CLOSE_RAD - _GRIPPER_OPEN_RAD) * progress
            # Measured before this step's command is applied, so it reflects the previous one; the
            # ramp moves far less than the contact threshold per step, so that costs nothing.
            achieved = float(robot.data.joint_pos[:, self._gripper_index].max())
            if achieved - command <= _GRIP_CONTACT_LAG_RAD:
                return command
            self._grip_hold = achieved - _GRIP_SQUEEZE_RAD
            self._grip_contact = (step, command)
            print(
                f"[LiftCubePickPlace] jaws stopped at gripper={math.degrees(achieved):.1f} deg with the command "
                f"at {math.degrees(command):.1f} deg; settling to {math.degrees(self._grip_hold):.1f} deg"
            )

        contact_step, contact_command = self._grip_contact
        return self._ease_scalar(contact_command, self._grip_hold, (step - contact_step) / _GRIP_SETTLE_STEPS)

    # Joint-space helpers
    # ------------------------------------------------------------------

    def _arm_command(self, env, values: dict[str, float | torch.Tensor]) -> torch.Tensor:
        """Assemble a ``(num_envs, 5)`` arm command from per-joint angles, clamped to the limits."""
        columns = []
        for name in self._arm_names:
            value = values[name]
            if not isinstance(value, torch.Tensor):
                value = torch.full((env.num_envs,), float(value), device=env.device)
            columns.append(value)
        return torch.clamp(torch.stack(columns, dim=-1), self._joint_limits[:, 0], self._joint_limits[:, 1])

    def _solve_arm(self, robot, jaw_target_w: torch.Tensor) -> torch.Tensor:
        """Arm command that puts the jaw at ``jaw_target_w`` with the gripper pointing down."""
        jaw_target_b = quat_apply(quat_inv(robot.data.root_quat_w), jaw_target_w - robot.data.root_pos_w)
        wrist_flex_limits = tuple(math.radians(value) for value in SO101_FOLLOWER_USD_JOINT_LIMLITS["wrist_flex"])
        solution = self._model.top_down_joints(jaw_target_b, wrist_flex_limits=wrist_flex_limits)
        values = {
            "shoulder_pan": solution[:, 0],
            "shoulder_lift": solution[:, 1],
            "elbow_flex": solution[:, 2],
            "wrist_flex": solution[:, 3],
            "wrist_roll": torch.full_like(solution[:, 0], math.radians(_WRIST_ROLL_DEG)),
        }
        columns = [values[name] for name in self._arm_names]
        return torch.clamp(torch.stack(columns, dim=-1), self._joint_limits[:, 0], self._joint_limits[:, 1])

    @staticmethod
    def _progress(step: int, start: int, end: int) -> float:
        return (step - start) / float(end - start)

    @staticmethod
    def _blend(start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
        """Ease from ``start`` to ``end`` in joint space."""
        return start + (end - start) * _ease(alpha)

    @staticmethod
    def _ease_scalar(start: float, end: float, alpha: float) -> float:
        """Ease a single joint angle. Used for the gripper, which is commanded as a scalar."""
        return start + (end - start) * _ease(alpha)

    def _log_phase_start(self, env, robot, step: int, arm: torch.Tensor) -> None:
        """DEBUG: report the commanded and achieved state at each phase boundary.

        Remove once the grasp and hold posture are confirmed correct in sim.
        """
        phase_name = next((name for boundary, name in _PHASE_START_NAMES if boundary == step), None)
        if phase_name is None:
            return
        jaw_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
        cube_pos_w = env.scene["cube"].data.root_pos_w
        commanded = {name: float(arm[0, idx]) for idx, name in enumerate(self._arm_names)}
        pitches = self._model.pitches_from_joints(
            commanded["shoulder_lift"], commanded["elbow_flex"], commanded["wrist_flex"]
        )
        measured = robot.data.joint_pos[0, self._arm_indices]
        print(
            f"[LiftCubePickPlace][step={step}] entering phase={phase_name!r} "
            f"cube_pos={cube_pos_w[0].tolist()} jaw_pos={jaw_pos_w[0].tolist()} "
            f"jaw_minus_cube={(jaw_pos_w[0] - cube_pos_w[0]).tolist()} "
            f"joints={self._arm_names} "
            f"cmd_deg={[round(math.degrees(float(value)), 1) for value in arm[0]]} "
            f"actual_deg={[round(math.degrees(float(value)), 1) for value in measured]} "
            f"cmd_link_pitch_deg={[round(math.degrees(wrap_to_pi(value)), 1) for value in pitches]} "
            f"jaw_heading_deg={math.degrees(self._model.finger_azimuth(commanded['shoulder_pan'])):.1f} "
            f"cube_heading_deg={math.degrees(self._cube_heading(env, robot)):.1f}"
        )

    @staticmethod
    def _cube_heading(env, robot) -> float:
        """Base-frame heading of the cube's own X axis, i.e. which way its faces point.

        Reported only, for now. The cube spawns with up to 30 deg of yaw, so the jaws can meet it
        corner-on rather than face-on; comparing this with ``jaw_heading_deg`` modulo 90 deg says
        by how much.
        """
        cube_quat_w = env.scene["cube"].data.root_quat_w
        axis = torch.zeros_like(cube_quat_w[:, :3])
        axis[:, 0] = 1.0
        axis_b = quat_apply(quat_inv(robot.data.root_quat_w), quat_apply(cube_quat_w, axis))
        return float(torch.atan2(axis_b[0, 1], axis_b[0, 0]))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count
