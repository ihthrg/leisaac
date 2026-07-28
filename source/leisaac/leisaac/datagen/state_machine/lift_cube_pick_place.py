"""State machine for the conditional lift-cube pick-and-place (sort) task."""

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS
from leisaac.tasks.lift_cube.mdp import cube_placed_on_correct_target

from .base import StateMachineBase
from .planar_arm_model import calibrate_planar_arm

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
be -- see ``_GRIP_STALL_FRACTION`` and ``_GRIP_SQUEEZE_RAD`` below."""

_CUBE_SIZE = 0.03
"""Edge length (m) of the cube this task picks up.

Measured from the asset, not assumed: the collider mesh in
``assets/scenes/table_with_cube/cube/`` is an eight-vertex box 30.15 mm on a side, and
``model.xml`` gives its region bbox as ``size="0.015077 ..."`` (MuJoCo box sizes are half
extents). Nothing scales it on the way into the scene, which the sim confirms -- the fingertip
came to rest 15.2 mm from the cube's centre along the cube's own X axis. The 0.15 mm rounded off
here is well inside the calibration's own residual."""

_CUBE_HALF_SPAN = 0.5 * _CUBE_SIZE * math.sqrt(2.0)
"""Half the cube's widest horizontal silhouette, i.e. across its diagonal.

The cube spawns with up to 30 deg of yaw and the jaws take no notice of it, so the half-width the
fingers actually meet is anywhere between ``_CUBE_SIZE / 2`` and this. Clearances have to assume
the worst of the two, or the descent puts a finger through a corner."""

_GRIP_STALL_FRACTION = 0.25
_GRIP_STALL_STEPS = 5
_GRIP_STALL_BLANKING_STEPS = 20
"""Stall test: the jaws have met something once the joint has closed by less than
``_GRIP_STALL_FRACTION`` of a step's worth of ramp for ``_GRIP_STALL_STEPS`` steps in a row,
ignoring the first ``_GRIP_STALL_BLANKING_STEPS`` while the joint is still picking up speed.

How far the joint *lags* its command was tried first and cannot be thresholded. That lag is
``rate * damping / stiffness`` plus whatever the jaws are pushing against, so it grows while the
fingers are still shoving the cube across the table, long before they pin it. In sim it tripped at
20.1, 20.6 and 20.7 deg on three different cube poses -- each time at exactly the threshold, which
is the signature of a lag that grew into it rather than of the cube stopping anything. How fast
the joint is still closing needs no such calibration: free it tracks the ramp, blocked it stops."""

_EMPTY_GRIP_MARGIN = math.radians(5.0)
"""How far clear of their own closed stop the jaws must halt for the cube to be in them.

The stall detector cannot tell "stopped by the cube" from "stopped by the other finger" -- both
simply stop the joint. Measured over 22 sim episodes the two are nowhere near each other: the jaws
shut on air halted between 2.6 and 3.1 deg, while every real grasp halted between 14.0 and
20.6 deg. This margin sits in that empty band, ten times the 0.5 deg spread of the closed stop and
still 9 deg below the tightest real grasp."""

_GRIP_SQUEEZE_RAD = 0.15
"""Interference held past the angle the cube stopped the jaws at.

The actuator is position controlled, so this sets the grip force: ``stiffness * this`` = 2.7 N m,
about 38 N at the finger's ~7 cm radius, well inside the 10 N m effort limit.

That is still far more than is needed to carry a light cube, and deliberately so. The cube is
grasped with up to 30 deg of yaw on it, so the jaws close on a corner rather than a face; a finger
bearing on a corner torques the cube towards lying flat against it, and the sim logs show exactly
that happening -- 23.7 deg of misalignment became 8.1 deg during the grasp. What they also show is
the rotation stopping part-way, leaving the cube cocked and free to slip out. The grip has to be
firm enough to finish turning the cube against table friction, not merely to hold it once turned.

Lowered from 0.20 rad (3.6 N m) on request. Grasps stall between 14.0 and 20.6 deg, and the clamp
at ``_GRIPPER_CLOSE_RAD`` never reaches that far, so the whole reduction lands on the grip: the
held command now sits 8.6 deg inside the stall instead of 11.5 deg, i.e. at 5.4 to 12.0 deg rather
than 2.5 to 9.1 deg.

The failure this risks is specific, and worth watching for rather than assuming away: a cube still
visibly cocked as it leaves the table, or one that slips out during the lift or the carry. Either
means the squeeze no longer finishes squaring it, and the number should go back up."""

_GRIP_SETTLE_STEPS = 30
"""Steps (0.5s) spent easing the command back from where contact was noticed to the squeeze."""

_GRASP_FACE_CLEARANCE = 0.005
"""Gap (m) left between the stationary finger and the cube while descending.

The tracked point sits ``_CUBE_HALF_SPAN + this`` from the stationary finger, rather than in the
middle of the jaws. Both keep the cube off the stationary finger on the way down, but the middle
is needlessly far: the moving finger then has to travel the whole way in before it touches, and
shoves the cube ahead of it until it pins. Measured in sim at the midpoint, that shove was 18 mm;
from here it is a few."""

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

_FINGER_TABLE_CLEARANCE = 0.008
"""Height (m) the lowest fingertip is kept above the table while gripping.

The descent aims the tracked point at ``table + this + finger_drop``, where ``finger_drop`` is
measured in ``setup()`` -- see ``_measure_finger_drop``. Fixing a depth below the cube's centre
instead, as earlier revisions did, assumes the tracked point sits at the same height as the
fingertips. It does not: the jaws sweep 57 deg, and a third of that motion is vertical, so the
tracked point rides about a centimetre above the fingers. In sim that put the fingertip 1 mm off
the table and the arm ended up standing on it -- after 3.5 s of a *static* grasp command,
``shoulder_lift`` was still 2.9 deg short of it, while the same arm tracks its commands exactly in
free air.

8 mm is generous against the calibration's own few-millimetre residual and costs nothing: the
fingers are far longer than the cube is tall, so they still straddle it from well below its
mid-height."""

_TABLE_PROBE_PAN_OFFSET_DEG = 25.0
_TABLE_PROBE_SETTLE_STEPS = 90
"""Table-contact probe: swing the grasp configuration this far about ``shoulder_pan`` so it meets
bare table instead of the cube, command it, and give it this long (1.5 s, nine actuator time
constants) to settle.

Panning is what makes the probe valid. It rotates the whole arm about a vertical axis, so every
link pitch -- and therefore every height, and therefore which part of the gripper is lowest -- is
identical to the real grasp; only the heading changes.

Driving the arm there rather than teleporting it is the other half. Teleporting places the arm
inside whatever is in the way and reports it as reached; driving it and reading where it *stops*
is what makes the obstacle measurable at all."""

_TABLE_BITE_WARN = 0.004
"""Table-probe reading (m) above which ``setup()`` warns rather than quietly lifting the grasp.

The grasp only has a few millimetres of room. The gripper's lowest point sits about 15 mm below
the point the model tracks, and the cube is 30 mm tall, so raising the grasp much beyond a
millimetre or two starts the moving finger above the cube's top face -- it then sweeps across the
top rather than down the side, and shoves the cube away instead of pinning it. A reading this
large means the probe is hitting something the grasp pose itself would not, and the number should
be investigated rather than trusted."""

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


def _ease(alpha: float) -> float:
    """Smoothstep easing, so each phase starts and ends at zero joint velocity.

    Plain linear interpolation would step the commanded velocity discontinuously at every phase
    boundary, which jolts a carried cube.
    """
    alpha = min(max(alpha, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


def _ease_tensor(alpha: torch.Tensor) -> torch.Tensor:
    """``_ease`` for a per-environment progress tensor.

    The gripper settles onto whatever each environment's jaws found, at whatever step they found
    it, so its easing cannot share the single scalar progress the arm phases use.
    """
    return 0.5 - 0.5 * torch.cos(math.pi * torch.clamp(alpha, 0.0, 1.0))


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
        self._grip_hold: torch.Tensor | None = None
        self._grip_latched: torch.Tensor | None = None
        self._grip_stall: torch.Tensor | None = None
        self._grip_contact_step: torch.Tensor | None = None
        self._grip_contact_command: torch.Tensor | None = None
        self._grip_last_pos: torch.Tensor | None = None
        self._grip_stalled_steps: torch.Tensor | None = None
        self._finger_drop: float = 0.0
        self._table_bite: float = 0.0
        self._empty_grip: float = 0.0
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

        def drive_pose(pose_deg: dict[str, float], steps: int) -> torch.Tensor:
            """Command a pose and let the arm *drive* to it, then report the arm joints it reached.

            Unlike ``write_pose`` this does not teleport. Teleporting puts the arm inside whatever
            is in the way and reports it as reached; driving it there and reading where it stops is
            the only way to find out that something stopped it.
            """
            joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
            for idx, name in enumerate(joint_names):
                joint_pos[:, idx] = math.radians(pose_deg.get(name, 0.0))
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()
            for _ in range(steps):
                env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            return robot.data.joint_pos[0, self._arm_indices].clone()

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
            grasp_offset=_CUBE_HALF_SPAN + _GRASP_FACE_CLEARANCE,
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
        self._finger_drop = self._measure_finger_drop(env, robot, probe)
        self._table_bite = self._measure_table_bite(env, robot, write_pose, drive_pose)
        self._empty_grip = self._measure_empty_grip(env, robot, write_pose, drive_pose)

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
            f"finger_drop={self._finger_drop * 1000.0:.1f} mm, table_bite={self._table_bite * 1000.0:.1f} mm, "
            f"empty_grip={math.degrees(self._empty_grip):.1f} deg, "
            f"center_x={self._center_x}"
        )
        if residual > _CALIB_RESIDUAL_WARN:
            print(
                "[LiftCubePickPlace][setup] WARNING: the measured kinematics disagree with the robot by "
                f"{residual * 1000.0:.1f} mm at the hold pose. Every jaw target will be off by roughly that "
                "much; check that the calibration probes are being held steady."
            )
        if self._table_bite > _TABLE_BITE_WARN:
            print(
                "[LiftCubePickPlace][setup] WARNING: the table probe wants the grasp raised "
                f"{self._table_bite * 1000.0:.1f} mm, far more than the millimetre or two it should take. The "
                "grasp is being lifted far enough that the moving finger may start above the cube and sweep "
                "it aside instead of pinning it; check what the probe is actually landing on."
            )

    def _measure_finger_drop(self, env, robot, probe) -> float:
        """How far (m) the lowest fingertip hangs below the point the model tracks.

        The model tracks a point part-way along the closed-to-open fingertip segment. That segment
        is far from horizontal -- the jaws sweep 57 deg, and a third of the fingertip's travel is
        vertical -- so the tracked point rides well above the fingers themselves. Aiming it at a
        fixed depth below the cube's centre therefore drives the fingers into the table, which is
        exactly what happened: the fingertip ended up 1 mm above the table and the arm stood on it.

        Measured, not derived, and measured in the pose it matters in: the arm is driven to the
        configuration the grasp actually uses, and the two fingertips are read there. Both are
        checked because which one is lower depends on the gripper angle, and the jaws pass through
        the whole range during the grasp.
        """
        target_w = env.scene["cube"].data.root_pos_w.clone()
        solution = self._solve_arm(robot, target_w)
        angles = {name: float(solution[0, idx]) for idx, name in enumerate(self._arm_names)}
        probe_deg = tuple(math.degrees(angles[name]) for name in _PROBE_JOINT_NAMES)

        closed_tip = probe(*probe_deg, _WRIST_ROLL_DEG, math.degrees(_GRIPPER_CLOSE_RAD))[0]
        open_tip = probe(*probe_deg, _WRIST_ROLL_DEG, math.degrees(_GRIPPER_OPEN_RAD))[0]
        tracked_z = closed_tip[2] + self._model.grasp_fraction * (open_tip[2] - closed_tip[2])
        return tracked_z - min(closed_tip[2], open_tip[2])

    def _measure_empty_grip(self, env, robot, write_pose, drive_pose) -> float:
        """Gripper angle (rad) the jaws come to rest at with nothing between them.

        The stall detector cannot tell "stopped by the cube" from "stopped by the other finger":
        both simply stop the joint, and in sim the second one was recorded as a successful pick
        three times. Measuring where the empty jaws stop gives ``check_success`` something to
        compare the grasp against.

        Driven rather than teleported, and in free air above the cube, for the same reason as the
        table probe: teleporting places the joint at the commanded angle and reports it as reached,
        which is precisely the reading that has to be avoided.
        """
        hover_w = env.scene["cube"].data.root_pos_w.clone()
        hover_w[:, 2] += _HOVER_CLEARANCE
        solution = self._solve_arm(robot, hover_w)
        pose = {name: math.degrees(float(solution[0, idx])) for idx, name in enumerate(self._arm_names)}
        write_pose({**pose, "gripper": math.degrees(_GRIPPER_OPEN_RAD)})
        # Same settle time as the table probe -- long enough for the joint to stop moving.
        drive_pose({**pose, "gripper": math.degrees(_GRIPPER_CLOSE_RAD)}, _TABLE_PROBE_SETTLE_STEPS)
        return float(robot.data.joint_pos[0, self._gripper_index])

    def _measure_table_bite(self, env, robot, write_pose, drive_pose) -> float:
        """How much (m) the grasp height has to rise for the gripper to stop touching the table.

        ``_measure_finger_drop`` can only see the frame Isaac Sim publishes, which follows the
        *moving* finger. Whatever the stationary finger, its knuckle or its mount extend below that
        point is invisible to it, and comes straight out of the table clearance: in sim, aiming the
        reported fingertip 8 mm above the table still left the arm held 1.2 mm short of a static
        grasp command, which is the signature of something touching down there.

        Nothing in the reported frames will ever reveal that geometry, so it is measured by running
        into it: the arm is driven to the grasp pose and however far it settles above its command is
        how far that pose has to rise.

        Probing **at the grasp height** is the whole point, and probing anywhere else is wrong.
        ``top_down_joints`` gives up gripper pitch where a vertical gripper is out of reach, so the
        arm's posture -- and therefore which part of the gripper hangs lowest -- depends on the
        height being asked for. Reading it 13 mm lower, with the reported fingertip at table height,
        returned 10.1 mm where the grasp pose itself only wanted 1.2 mm. Lifting the grasp by that
        put the moving finger above the cube: it swept across the top instead of down the side,
        shoved the cube 14.5 mm, flipped it 19.4 mm into the air and closed on nothing. The window
        is narrow -- a 30 mm cube, and a gripper whose lowest point sits 15 mm below the tracked
        one -- so the correction has to be exactly the height's own, not another pose's.

        The probe is swung sideways about ``shoulder_pan`` so it lands on bare table rather than on
        the cube. Panning rotates the arm about a vertical axis, leaving every link pitch -- and so
        the whole vertical geometry -- exactly as the grasp will have it.

        Both gripper angles are tried and the worse taken, because which part hangs lowest changes
        as the jaws close and the grasp passes through the whole range.
        """
        cube_pos_w = env.scene["cube"].data.root_pos_w
        hover_w = cube_pos_w.clone()
        hover_w[:, 2] += _HOVER_CLEARANCE
        # Probed at the height the grasp will actually use, so the reading is the correction that
        # height needs and nothing else.
        touch_w = cube_pos_w.clone()
        touch_w[:, 2] += self._finger_drop + _FINGER_TABLE_CLEARANCE - 0.5 * _CUBE_SIZE

        pan_index = self._arm_names.index("shoulder_pan")
        offset = math.radians(_TABLE_PROBE_PAN_OFFSET_DEG)
        if float(self._solve_arm(robot, touch_w)[0, pan_index]) + offset > float(self._joint_limits[pan_index, 1]):
            offset = -offset

        def swung(target_w: torch.Tensor, gripper_deg: float) -> dict[str, float]:
            solution = self._solve_arm(robot, target_w)
            pose = {name: math.degrees(float(solution[0, idx])) for idx, name in enumerate(self._arm_names)}
            pose["shoulder_pan"] += math.degrees(offset)
            pose["gripper"] = gripper_deg
            return pose

        def height_of(arm_row: torch.Tensor) -> float:
            angles = {name: float(arm_row[idx]) for idx, name in enumerate(self._arm_names)}
            return self._model.jaw_in_plane(angles["shoulder_lift"], angles["elbow_flex"], angles["wrist_flex"])[1]

        bite = 0.0
        for gripper_deg in (math.degrees(_GRIPPER_OPEN_RAD), math.degrees(_GRIPPER_CLOSE_RAD)):
            # Start clear of the table so the arm comes down onto it instead of being placed inside
            # it, then command the touching pose and read where it actually stops.
            write_pose(swung(hover_w, gripper_deg))
            commanded = swung(touch_w, gripper_deg)
            achieved = drive_pose(commanded, _TABLE_PROBE_SETTLE_STEPS)
            wanted = torch.tensor([math.radians(commanded[name]) for name in self._arm_names], device=achieved.device)
            bite = max(bite, height_of(achieved) - height_of(wanted))
        return bite

    def check_success(self, env) -> torch.Tensor:
        """Return, for each environment, whether the cube was picked up and left on its target.

        The placement test alone is not enough. In sim it passed three times on episodes whose
        jaws shut on empty air -- the cube reached the marker without ever being gripped -- and
        those were recorded as demonstrations of picking it up. So the grasp has to be shown as
        well: the jaws must have stalled, and stalled clear of the point they stop at when there
        is nothing between them.
        """
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

        held = torch.logical_and(self._grip_latched, self._grip_stall > self._empty_grip + _EMPTY_GRIP_MARGIN)
        for env_id in torch.logical_and(torch.logical_and(placed, at_rest), ~held).nonzero().flatten().tolist():
            print(
                f"[LiftCubePickPlace][env={env_id}] the cube ended up on its target, but the jaws stalled at "
                f"{math.degrees(float(self._grip_stall[env_id])):.1f} deg against an empty-jaw stop of "
                f"{math.degrees(self._empty_grip):.1f} deg, so nothing was ever gripped. Not recording it."
            )
        return torch.logical_and(torch.logical_and(placed, at_rest), held)

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
            self._grip_hold = torch.full((env.num_envs,), _GRIPPER_CLOSE_RAD, device=env.device)
            self._grip_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            self._grip_stall = torch.full((env.num_envs,), _GRIPPER_CLOSE_RAD, device=env.device)
            self._grip_contact_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            self._grip_contact_command = torch.full((env.num_envs,), _GRIPPER_CLOSE_RAD, device=env.device)
            self._grip_stalled_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            self._grip_last_pos = None

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
        # Referenced to the table the cube sits on, not to the cube's centre, because what must
        # not collide is the gripper's lowest point and what it must not collide with is the table.
        # Both offsets are measured in ``setup()``: what the reported fingertip clears, and what it
        # does not report at all.
        grasp = raised(
            self._cube_pos_w,
            self._finger_drop + self._table_bite + _FINGER_TABLE_CLEARANCE - 0.5 * _CUBE_SIZE,
        )
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

        # Phases that do not touch the cube command one angle for every environment; the ones that
        # do return a value per environment.
        if not isinstance(gripper, torch.Tensor):
            gripper = torch.full((env.num_envs,), float(gripper), device=env.device)
        return torch.cat([arm, gripper.unsqueeze(-1)], dim=-1)

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
        self._grip_latched = None
        self._grip_stall = None
        self._grip_contact_step = None
        self._grip_contact_command = None
        self._grip_last_pos = None
        self._grip_stalled_steps = None

    # ------------------------------------------------------------------

    @property
    def _grip_command(self) -> torch.Tensor:
        """Gripper command to hold while the cube is being carried, one value per environment.

        Environments that never felt anything keep the fully-shut command they were initialised
        with, which leaves their jaws closed on empty air -- the honest outcome for an episode
        that missed the cube, and one the success check will reject on its own.
        """
        return self._grip_hold

    def _closing_gripper(self, robot, step: int) -> torch.Tensor:
        """Ramp the jaws shut, stop at whatever they land on, and hold a firm squeeze on it.

        The ramp is linear rather than eased so the joint's free-running speed -- which is what
        contact is judged against -- stays constant instead of varying through the phase.

        Contact is only noticed once the joint has been standing still for a few steps, by which
        time the command has already been driven past it, so the command is walked on to the
        squeeze over ``_GRIP_SETTLE_STEPS`` rather than jumped there, which would snatch at the
        cube just as it is being taken hold of.

        Every environment is judged on its own. Only the cube's pose is randomised, but that is
        enough to move the stall point by a few degrees between environments, and a shared command
        would hand all of them whichever jaws happened to block first.
        """
        elapsed = step - _LOWER_TO_CUBE_END
        rate = (_GRIPPER_OPEN_RAD - _GRIPPER_CLOSE_RAD) / _GRASP_CLOSE_STEPS
        command = _GRIPPER_OPEN_RAD - rate * min(elapsed, _GRASP_CLOSE_STEPS)

        # Measured before this step's command is applied, so it reflects the previous one.
        achieved = robot.data.joint_pos[:, self._gripper_index]
        closed_by = achieved if self._grip_last_pos is None else self._grip_last_pos - achieved
        self._grip_last_pos = achieved.clone()

        if elapsed < _GRIP_STALL_BLANKING_STEPS:
            self._grip_stalled_steps = torch.zeros_like(self._grip_stalled_steps)
        else:
            self._grip_stalled_steps = torch.where(
                closed_by > _GRIP_STALL_FRACTION * rate,
                torch.zeros_like(self._grip_stalled_steps),
                self._grip_stalled_steps + 1,
            )

        stalled = torch.logical_and(~self._grip_latched, self._grip_stalled_steps >= _GRIP_STALL_STEPS)
        if bool(stalled.any()):
            hold = torch.clamp(achieved - _GRIP_SQUEEZE_RAD, min=_GRIPPER_CLOSE_RAD)
            self._grip_hold = torch.where(stalled, hold, self._grip_hold)
            self._grip_stall = torch.where(stalled, achieved, self._grip_stall)
            self._grip_contact_step = torch.where(
                stalled, torch.full_like(self._grip_contact_step, step), self._grip_contact_step
            )
            self._grip_contact_command = torch.where(
                stalled, torch.full_like(self._grip_contact_command, command), self._grip_contact_command
            )
            self._grip_latched = torch.logical_or(self._grip_latched, stalled)
            for env_id in stalled.nonzero(as_tuple=False).flatten().tolist():
                print(
                    f"[LiftCubePickPlace][env={env_id}] jaws stalled at "
                    f"gripper={math.degrees(float(achieved[env_id])):.1f} deg with the command at "
                    f"{math.degrees(command):.1f} deg; squeezing to "
                    f"{math.degrees(float(self._grip_hold[env_id])):.1f} deg"
                )

        settling = _ease_tensor((step - self._grip_contact_step) / _GRIP_SETTLE_STEPS)
        eased = self._grip_contact_command + (self._grip_hold - self._grip_contact_command) * settling
        return torch.where(self._grip_latched, eased, torch.full_like(eased, command))

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count
