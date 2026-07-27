"""State machine for the conditional lift-cube pick-and-place (sort) task."""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul
from leisaac.tasks.lift_cube.mdp import cube_placed_on_correct_target

from .base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0

_REST_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 90.0,
    "wrist_flex": 50.0,
    "wrist_roll": 90.0,  # counter-clockwise 90 deg (was 0.0); flip the sign if this turns out
    # to be clockwise instead once verified in sim.
    "gripper": -10.0,
}

_MIDDLE_POSE_DEG = {
    "shoulder_pan": 0.0,  # motor ID 1
    "shoulder_lift": 90.0,  # motor ID 2
    "elbow_flex": 90.0,  # motor ID 3
    "wrist_flex": 0.0,  # motor ID 4
    "wrist_roll": 90.0,  # motor ID 5
    "gripper": 0.0,  # motor ID 6 (overridden by the binary gripper command while holding)
}
"""Joint pose used as the in-air hold target ("middle position").

This used to be the robot's ``init_state`` pose (all joints at 0 deg except ``wrist_roll``), but
that configuration zeroes all three bending joints at once, so the arm ends up fully extended
forward. A straight arm is a kinematic singularity, where the differential-IK solver used by this
task loses a DoF and converges poorly, so it is a bad pose to command. Bending ``shoulder_lift``
and ``elbow_flex`` to 90 deg keeps the cube in front of the camera while staying well away from
that singularity.

Note that ``elbow_flex`` at 90 deg sits exactly on its USD upper limit (see
``SO101_FOLLOWER_USD_JOINT_LIMLITS``), which is also what ``_REST_POSE_DEG`` uses, so the IK
solver cannot overshoot in that direction while tracking this pose."""

_GRASP_APPROACH_DIR_WORLD = (0.0, 0.0, -1.0)
"""World direction the gripper-to-jaw axis must point while grasping and placing. Pointing it
straight down makes the arm reach over the cube and close on it from above; these phases
previously reused the middle pose's orientation, which approached the cube from the side."""

_REST_POSE_TOLERANCE_DEG = 30.0
"""Per-joint +/- tolerance (deg) used by ``_is_at_rest`` to decide if the arm reached
``_REST_POSE_DEG``. Matches the tolerance used by the shared
``leisaac.utils.robot_utils.is_so101_at_rest_pose``/``SO101_FOLLOWER_REST_POSE_RANGE`` for every
joint except ``wrist_roll`` -- this task holds ``wrist_roll`` at 90 deg (not 0 deg) at rest, so it
uses its own local check instead of the shared one, to avoid shifting the shared rest-pose
tolerance used by other tasks."""

_HOLD_HEIGHT_ABOVE_TABLE = 0.20
"""Height (m) above the cube while lifting/carrying/holding it, expressed as a target for the
*jaw* (see below), not the raw IK control frame. The "middle position" hold target itself is
FK-calibrated in ``setup()``, not derived from this constant."""

_HOVER_CLEARANCE = 0.10
"""Jaw height (m) above the cube when hovering, before/after lowering to grasp."""

_RELEASE_CLEARANCE = 0.02
"""Jaw height (m) above the target marker when lowering to release the cube."""

# Phase boundaries (state-machine step count). Each `env.step()` advances physics by one sim
# tick, so a 60Hz simulation (`--step_hz 60`) requires `round(seconds * 60)` steps for the
# intended physical duration. `LeRobotRecorderManager` separately downsamples those steps to the
# requested dataset rate (`--lerobot_dataset_fps 30`, i.e. every second step here).
_APPROACH_STEPS = 120  # phase ends at 120 (2.0s): interpolate from initial EE pos to cube hover
_LOWER_TO_CUBE_END = 180  # +60 (1.0s)
_GRASP_END = 240  # +60 (1.0s)
_LIFT_CUBE_END = 300  # +60 (1.0s)
_MOVE_TO_MIDDLE_END = 390  # +90 (1.5s)
_HOLD_MIDDLE_END = 570  # +180 (3.0s) <- user requirement: hold ~3s near the "middle position"
_MOVE_ABOVE_TARGET_END = 660  # +90 (1.5s)
_LOWER_TO_TARGET_END = 720  # +60 (1.0s)
_RELEASE_END = 760  # +40 (0.67s)
_LIFT_GRIPPER_END = 800  # +40 (0.67s)
_TOTAL_STEPS = 1100  # +300 (5.0s): return home, ends the episode

# DEBUG: (step, phase-name) pairs used by the diagnostic print in get_action() to report which
# phase is starting and the key positions involved -- remove once the grasp/pose issues are
# resolved and confirmed fixed in sim.
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


def _quat_nlerp(start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate unit quaternions along the shortest path using normalized lerp."""
    end_shortest = torch.where(torch.sum(start * end, dim=-1, keepdim=True) < 0.0, -end, end)
    interpolated = start + (end_shortest - start) * alpha
    return interpolated / torch.linalg.vector_norm(interpolated, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_between_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shortest-arc quaternion rotating ``source`` onto ``target`` (both ``(N, 3)`` world vectors)."""
    source = source / torch.linalg.vector_norm(source, dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1.0e-8)

    cos_angle = torch.sum(source * target, dim=-1, keepdim=True)
    quat = torch.cat([1.0 + cos_angle, torch.cross(source, target, dim=-1)], dim=-1)

    # Exactly antiparallel vectors have no unique shortest arc; any perpendicular axis gives the
    # required 180 deg rotation, so pick whichever reference axis is not parallel to ``source``.
    primary_ref = torch.zeros_like(source)
    primary_ref[..., 0] = 1.0
    secondary_ref = torch.zeros_like(source)
    secondary_ref[..., 1] = 1.0
    perpendicular = torch.cross(source, primary_ref, dim=-1)
    perpendicular = torch.where(
        torch.linalg.vector_norm(perpendicular, dim=-1, keepdim=True) < 1.0e-6,
        torch.cross(source, secondary_ref, dim=-1),
        perpendicular,
    )
    quat = torch.where(
        cos_angle < -1.0 + 1.0e-6,
        torch.cat([torch.zeros_like(cos_angle), perpendicular], dim=-1),
        quat,
    )
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _azimuth(pos_w: torch.Tensor, base_pos_w: torch.Tensor) -> torch.Tensor:
    """Heading (rad) from the robot base to ``pos_w``, measured in the world XY plane."""
    delta = pos_w - base_pos_w
    return torch.atan2(delta[:, 1], delta[:, 0])


def _yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    """Quaternion for a rotation of ``angle`` (rad, shape ``(N,)``) about the world Z axis."""
    half = 0.5 * angle
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


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

    Picks up the cube, holds it for ~3 seconds at the SO-101 "middle position" (see
    ``_MIDDLE_POSE_DEG``), then places it on whichever target matches its original spawn side:
    the circle target if the cube spawned to the right (+X) of the nominal center, or the
    square target if it spawned to the left. Finally returns the arm to the SO-101 rest pose.

    Each episode starts by snapping the arm to the rest pose (see ``pre_step()``) before any
    action is recorded, so every recorded episode begins from a consistent rest state.
    """

    TOTAL_STEPS: int = _TOTAL_STEPS

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._initial_ee_pos: torch.Tensor | None = None
        self._initial_ee_quat_world: torch.Tensor | None = None
        self._rest_ee_pos_world: torch.Tensor | None = None
        self._rest_ee_quat_world: torch.Tensor | None = None
        self._rest_joint_pos: torch.Tensor | None = None
        self._home_start_pos: torch.Tensor | None = None
        self._hold_pos_world: torch.Tensor | None = None
        self._hold_quat_world: torch.Tensor | None = None
        self._grasp_quat_world: torch.Tensor | None = None
        self._grasp_azimuth_ref: torch.Tensor | None = None
        self._center_x: float = 0.0
        self._target_is_circle: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        """FK calibration for the rest and middle poses; other one-time setup.

        Teleports the joints to the SO-101 rest pose, then to the all-zero pose, then to the
        requested middle pose (see ``_MIDDLE_POSE_DEG``), reading the corresponding world-space
        gripper/jaw poses via FK. These are used as ordinary IK targets during the episode, so
        the arm reaches them through smooth, physically-simulated motion rather than teleporting
        while carrying the cube. The all-zero probe is only used to measure the arm's reach
        heading (see ``_grasp_azimuth_ref`` below); it is never commanded.

        The caller resets the environment after ``setup()``, so none of these calibration poses
        leak into the first recorded episode.

        ``center_x`` (the left/right decision threshold) is derived from the cube's *nominal*
        (pre-randomization) spawn point, i.e. ``env.cfg.scene.cube.init_state.pos`` -- the same
        fixed value the env config uses to place the two targets and to seed ``center_x`` in
        ``mdp.cube_placed_on_correct_target``/``cube_pick_place_done``. Reading this from the
        config (rather than the live, possibly-already-randomized ``root_pos_w``) guarantees the
        state machine's target choice always agrees with the success-check's.
        """
        robot = env.scene["robot"]
        joint_names = list(robot.data.joint_names)

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _REST_POSE_DEG:
                self._rest_joint_pos[:, idx] = _REST_POSE_DEG[name] * torch.pi / 180.0

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        # Use the ee_frame's "gripper" target (index 0, no offset) rather than
        # `robot.data.body_pos_w[:, -1, :]` -- the latter is whatever body happens to be *last*
        # in the articulation's body list, which is not guaranteed to be "gripper" if e.g. "jaw"
        # is a separate body ordered after it. This keeps every EE-space position in this file
        # anchored to the exact same, explicitly-named prim that the IK action also targets.
        ee_frame = env.scene["ee_frame"]
        self._rest_ee_pos_world = ee_frame.data.target_pos_w[:, 0, :].clone()
        self._rest_ee_quat_world = ee_frame.data.target_quat_w[:, 0, :].clone()

        # Reference heading for `_top_down_quat`: the horizontal direction the arm reaches when
        # `shoulder_pan` is 0. Measured with a dedicated FK probe at the all-zero joint pose (a
        # fully extended arm) instead of reusing the hold pose, because the hold pose is folded
        # back toward the base, where the base-to-jaw azimuth is ill-conditioned and can even be
        # flipped by 180 deg. The extended arm puts the jaw unambiguously far out in front, so
        # its azimuth is a reliable stand-in for "shoulder_pan == 0".
        zero_joint_pos = torch.zeros_like(self._rest_joint_pos)
        robot.write_joint_state_to_sim(
            position=zero_joint_pos,
            velocity=torch.zeros_like(zero_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._grasp_azimuth_ref = _azimuth(ee_frame.data.target_pos_w[:, 1, :], robot.data.root_pos_w)

        middle_joint_pos = torch.zeros_like(self._rest_joint_pos)
        for idx, name in enumerate(joint_names):
            if name in _MIDDLE_POSE_DEG:
                middle_joint_pos[:, idx] = _MIDDLE_POSE_DEG[name] * torch.pi / 180.0

        robot.write_joint_state_to_sim(
            position=middle_joint_pos,
            velocity=torch.zeros_like(middle_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._hold_pos_world = ee_frame.data.target_pos_w[:, 1, :].clone()
        self._hold_quat_world = ee_frame.data.target_quat_w[:, 0, :].clone()

        # Top-down grasp orientation: rotate the middle orientation by the shortest arc that
        # brings its gripper-to-jaw axis onto straight down, so the jaw closes on the cube from
        # above while otherwise staying as close to the middle orientation as possible.
        middle_approach_dir = ee_frame.data.target_pos_w[:, 1, :] - ee_frame.data.target_pos_w[:, 0, :]
        down_dir = torch.tensor(_GRASP_APPROACH_DIR_WORLD, device=env.device).expand_as(middle_approach_dir)
        self._grasp_quat_world = quat_mul(
            _quat_between_vectors(middle_approach_dir, down_dir),
            self._hold_quat_world,
        )

        nominal_cube_pos = env.cfg.scene.cube.init_state.pos
        self._center_x = float(nominal_cube_pos[0])

    def check_success(self, env) -> bool:
        """Return True if the cube is on its correct target and the arm is at rest.

        The placement check is evaluated *before* forcing the rest pose below, since it also
        inspects the current gripper joint angle (must be open, i.e. the cube was released) --
        teleporting to the rest pose first would spuriously report the gripper as closed.
        """
        placed = cube_placed_on_correct_target(
            env,
            cube_cfg=SceneEntityCfg("cube"),
            target_cfg=SceneEntityCfg("target"),
            circle_target_cfg=SceneEntityCfg("circle_target"),
            robot_cfg=SceneEntityCfg("robot"),
            center_x=self._center_x,
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
        """Snap to rest before the first step, then blend back to rest at the end.

        The middle target was calibrated once in ``setup()`` from the explicit middle joint
        pose. The initial snap ensures every recorded episode starts from a consistent rest
        state, matching the direct joint-state-write technique used at the end.
        """
        robot = env.scene["robot"]
        if self._step_count == 0 and self._rest_joint_pos is not None:
            robot.write_joint_state_to_sim(
                position=self._rest_joint_pos,
                velocity=torch.zeros_like(self._rest_joint_pos),
            )
            env.scene.update(dt=env.physics_dt)
        if self._step_count >= _LIFT_GRIPPER_END and self._rest_joint_pos is not None:
            if self._step_count == _LIFT_GRIPPER_END:
                self._home_start_pos = robot.data.joint_pos.clone()
            if self._home_start_pos is not None:
                alpha = min((self._step_count - _LIFT_GRIPPER_END) / float(_TOTAL_STEPS - _LIFT_GRIPPER_END - 1), 1.0)
                blended = self._home_start_pos + (self._rest_joint_pos - self._home_start_pos) * alpha
                robot.write_joint_state_to_sim(position=blended, velocity=torch.zeros_like(blended))

    def get_action(self, env) -> torch.Tensor:
        """Compute the action tensor for the current step (8D IK pose target).

        Cube/target positions below describe where the *jaw* (fingertip contact point, see
        ``ee_frame``'s second target frame) should be -- not the raw "gripper" IK control frame,
        which sits at a fixed but non-trivial offset from the jaw. The rigid offset is measured
        live, expressed in the gripper's local frame, then rotated by the *target* gripper
        orientation before being subtracted from every jaw target. This is necessary because
        using the current world-space offset is only correct when current and target
        orientations already match.

        The target orientation is scheduled per phase by ``_target_orientation()``: grasping and
        placing point the jaw straight down and yaw it toward whichever object is being reached,
        the mid-air hold uses the middle orientation, and return-home restores the rest
        orientation, with smooth interpolation in between.
        """
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs
        step = self._step_count

        cube_pos_w = env.scene["cube"].data.root_pos_w.clone()
        target_pos_w = env.scene["target"].data.root_pos_w.clone()
        circle_target_pos_w = env.scene["circle_target"].data.root_pos_w.clone()
        robot_base_pos_w = robot.data.root_pos_w.clone()
        robot_base_quat_w = robot.data.root_quat_w.clone()

        ee_frame = env.scene["ee_frame"]
        current_gripper_quat_w = ee_frame.data.target_quat_w[:, 0, :].clone()

        if step == 0:
            self._initial_ee_pos = ee_frame.data.target_pos_w[:, 0, :].clone()
            self._initial_ee_quat_world = current_gripper_quat_w.clone()
            self._target_is_circle = cube_pos_w[:, 0] > self._center_x

        selected_target_pos_w = torch.where(self._target_is_circle.unsqueeze(-1), circle_target_pos_w, target_pos_w)

        target_quat_w = self._target_orientation(
            step,
            current_gripper_quat_w,
            robot_base_pos_w=robot_base_pos_w,
            cube_pos_w=cube_pos_w,
            place_pos_w=selected_target_pos_w,
        )
        target_quat = quat_mul(quat_inv(robot_base_quat_w), target_quat_w)

        gripper_to_jaw_world = ee_frame.data.target_pos_w[:, 1, :] - ee_frame.data.target_pos_w[:, 0, :]
        gripper_to_jaw_local = quat_apply(quat_inv(current_gripper_quat_w), gripper_to_jaw_world)
        gripper_to_jaw = quat_apply(target_quat_w, gripper_to_jaw_local)

        # DEBUG: report which phase is starting and the key positions involved (env 0 only) --
        # remove once the grasp/pose issues are resolved and confirmed fixed in sim.
        for boundary_step, phase_name in _PHASE_START_NAMES:
            if step == boundary_step:
                print(
                    f"[LiftCubePickPlace][step={step}] entering phase={phase_name!r} "
                    f"cube_pos={cube_pos_w[0].tolist()} jaw_pos={ee_frame.data.target_pos_w[0, 1, :].tolist()} "
                    f"gripper_to_jaw={gripper_to_jaw[0].tolist()} "
                    f"current_quat_w={current_gripper_quat_w[0].tolist()} "
                    f"target_quat_w={target_quat_w[0].tolist()} "
                    f"hold_pos_world={self._hold_pos_world[0].tolist() if self._hold_pos_world is not None else None} "
                    f"rest_ee_pos_world={self._rest_ee_pos_world[0].tolist() if self._rest_ee_pos_world is not None else None}"
                )
                break

        if step < _APPROACH_STEPS:
            pos_w, gripper_cmd = self._phase_approach_hover(cube_pos_w, gripper_to_jaw, num_envs, device)
        elif step < _LOWER_TO_CUBE_END:
            pos_w, gripper_cmd = self._phase_lower_to_cube(cube_pos_w, gripper_to_jaw, num_envs, device)
        elif step < _GRASP_END:
            pos_w, gripper_cmd = self._phase_grasp(cube_pos_w, gripper_to_jaw, num_envs, device)
        elif step < _LIFT_CUBE_END:
            pos_w, gripper_cmd = self._phase_lift_cube(cube_pos_w, gripper_to_jaw, num_envs, device)
        elif step < _MOVE_TO_MIDDLE_END:
            pos_w, gripper_cmd = self._phase_move_to_middle(cube_pos_w, gripper_to_jaw, num_envs, device)
        elif step < _HOLD_MIDDLE_END:
            pos_w, gripper_cmd = self._phase_hold_middle(gripper_to_jaw, num_envs, device)
        elif step < _MOVE_ABOVE_TARGET_END:
            pos_w, gripper_cmd = self._phase_move_above_target(selected_target_pos_w, gripper_to_jaw, num_envs, device)
        elif step < _LOWER_TO_TARGET_END:
            pos_w, gripper_cmd = self._phase_lower_to_target(selected_target_pos_w, gripper_to_jaw, num_envs, device)
        elif step < _RELEASE_END:
            pos_w, gripper_cmd = self._phase_release(selected_target_pos_w, gripper_to_jaw, num_envs, device)
        elif step < _LIFT_GRIPPER_END:
            pos_w, gripper_cmd = self._phase_lift_gripper(selected_target_pos_w, gripper_to_jaw, num_envs, device)
        else:
            pos_w, gripper_cmd = self._phase_return_home(num_envs, device)

        diff_w = pos_w - robot_base_pos_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)
        return torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)

    def advance(self) -> None:
        """Advance the step counter; the episode ends once the return-home phase completes."""
        self._step_count += 1
        if self._step_count >= _TOTAL_STEPS:
            self._episode_done = True

    def reset(self) -> None:
        """Reset the state machine to its initial state for a new episode."""
        self._step_count = 0
        self._episode_done = False
        self._initial_ee_pos = None
        self._initial_ee_quat_world = None
        self._home_start_pos = None
        self._target_is_circle = None

    # ------------------------------------------------------------------
    # Orientation schedule
    # ------------------------------------------------------------------

    def _top_down_quat(self, pos_w: torch.Tensor, robot_base_pos_w: torch.Tensor) -> torch.Tensor:
        """Top-down orientation yawed toward ``pos_w`` as seen from the robot base.

        The IK action drives only five joints (``shoulder_pan`` plus three pitch joints and
        ``wrist_roll``) while the command is a full 6-DoF pose, so an orientation whose yaw does
        not match the arm's own swing plane is unreachable and the solver trades away position
        accuracy to chase it -- which is what made the gripper close on empty air. Rotating about
        the world Z axis leaves the jaw pointing straight down while matching the heading
        ``shoulder_pan`` produces, keeping the command inside the reachable set.
        """
        delta_yaw = _azimuth(pos_w, robot_base_pos_w) - self._grasp_azimuth_ref
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))  # wrap to [-pi, pi]
        return quat_mul(_yaw_quat(delta_yaw), self._grasp_quat_world)

    def _target_orientation(
        self,
        step: int,
        current_gripper_quat_w: torch.Tensor,
        robot_base_pos_w: torch.Tensor,
        cube_pos_w: torch.Tensor,
        place_pos_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return the world-space gripper orientation commanded at ``step``.

        Picking the cube up and releasing it onto a marker both use the top-down orientation
        yawed toward that object, so the jaw descends onto it rather than reaching in sideways.
        Only the mid-air hold uses the middle orientation, which is left exactly as calibrated so
        the hold reproduces the middle pose, and the episode ends back at the rest orientation.
        """
        if self._grasp_quat_world is None or self._hold_quat_world is None or self._grasp_azimuth_ref is None:
            return current_gripper_quat_w

        cube_quat = self._top_down_quat(cube_pos_w, robot_base_pos_w)
        place_quat = self._top_down_quat(place_pos_w, robot_base_pos_w)

        if step < _APPROACH_STEPS:
            start_quat = (
                self._initial_ee_quat_world if self._initial_ee_quat_world is not None else current_gripper_quat_w
            )
            return _quat_nlerp(start_quat, cube_quat, step / float(_APPROACH_STEPS))
        if step < _LIFT_CUBE_END:
            return cube_quat
        if step < _MOVE_TO_MIDDLE_END:
            alpha = (step - _LIFT_CUBE_END) / float(_MOVE_TO_MIDDLE_END - _LIFT_CUBE_END)
            return _quat_nlerp(cube_quat, self._hold_quat_world, alpha)
        if step < _HOLD_MIDDLE_END:
            return self._hold_quat_world.clone()
        if step < _MOVE_ABOVE_TARGET_END:
            alpha = (step - _HOLD_MIDDLE_END) / float(_MOVE_ABOVE_TARGET_END - _HOLD_MIDDLE_END)
            return _quat_nlerp(self._hold_quat_world, place_quat, alpha)
        if step < _LIFT_GRIPPER_END:
            return place_quat
        if self._rest_ee_quat_world is not None:
            return self._rest_ee_quat_world.clone()
        return current_gripper_quat_w

    # ------------------------------------------------------------------
    # Phase methods
    #
    # Each phase (other than the final return-home) computes a *jaw* target position; the
    # `gripper_to_jaw` (world-frame) offset passed in is subtracted to get the actual IK
    # ("gripper" frame) target sent to the arm.
    # ------------------------------------------------------------------

    def _phase_approach_hover(self, cube_pos_w, gripper_to_jaw, num_envs, device):
        hover_jaw = cube_pos_w.clone()
        hover_jaw[:, 2] += _HOVER_CLEARANCE
        hover_gripper_target = hover_jaw - gripper_to_jaw
        alpha = self._step_count / _APPROACH_STEPS
        if self._initial_ee_pos is not None:
            pos_w = (1.0 - alpha) * self._initial_ee_pos + alpha * hover_gripper_target
        else:
            pos_w = hover_gripper_target
        return pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_lower_to_cube(self, cube_pos_w, gripper_to_jaw, num_envs, device):
        jaw_target = cube_pos_w.clone()  # jaw at the cube's center
        return jaw_target - gripper_to_jaw, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_grasp(self, cube_pos_w, gripper_to_jaw, num_envs, device):
        jaw_target = cube_pos_w.clone()
        return jaw_target - gripper_to_jaw, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_lift_cube(self, cube_pos_w, gripper_to_jaw, num_envs, device):
        jaw_target = cube_pos_w.clone()
        jaw_target[:, 2] += _HOLD_HEIGHT_ABOVE_TABLE
        return jaw_target - gripper_to_jaw, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_move_to_middle(self, cube_pos_w, gripper_to_jaw, num_envs, device):
        lift_jaw = cube_pos_w.clone()
        lift_jaw[:, 2] += _HOLD_HEIGHT_ABOVE_TABLE
        alpha = (self._step_count - _LIFT_CUBE_END) / float(_MOVE_TO_MIDDLE_END - _LIFT_CUBE_END)
        jaw_target = (1.0 - alpha) * lift_jaw + alpha * self._hold_pos_world
        return jaw_target - gripper_to_jaw, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_hold_middle(self, gripper_to_jaw, num_envs, device):
        jaw_target = self._hold_pos_world.clone()
        return jaw_target - gripper_to_jaw, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_move_above_target(self, selected_target_pos_w, gripper_to_jaw, num_envs, device):
        target_hover = selected_target_pos_w.clone()
        target_hover[:, 2] += _HOLD_HEIGHT_ABOVE_TABLE
        alpha = (self._step_count - _HOLD_MIDDLE_END) / float(_MOVE_ABOVE_TARGET_END - _HOLD_MIDDLE_END)
        jaw_target = (1.0 - alpha) * self._hold_pos_world + alpha * target_hover
        return jaw_target - gripper_to_jaw, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_lower_to_target(self, selected_target_pos_w, gripper_to_jaw, num_envs, device):
        jaw_target = selected_target_pos_w.clone()
        jaw_target[:, 2] += _RELEASE_CLEARANCE
        return jaw_target - gripper_to_jaw, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_release(self, selected_target_pos_w, gripper_to_jaw, num_envs, device):
        jaw_target = selected_target_pos_w.clone()
        jaw_target[:, 2] += _RELEASE_CLEARANCE
        return jaw_target - gripper_to_jaw, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_lift_gripper(self, selected_target_pos_w, gripper_to_jaw, num_envs, device):
        jaw_target = selected_target_pos_w.clone()
        jaw_target[:, 2] += _HOLD_HEIGHT_ABOVE_TABLE
        return jaw_target - gripper_to_jaw, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_return_home(self, num_envs, device):
        if self._rest_ee_pos_world is not None:
            pos_w = self._rest_ee_pos_world.clone()
        elif self._initial_ee_pos is not None:
            pos_w = self._initial_ee_pos.clone()
        else:
            pos_w = torch.zeros(num_envs, 3, device=device)
        return pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count
