"""State machine for the conditional lift-cube pick-and-place (sort) task."""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul
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
computed via FK calibration (see ``setup()``), not from this constant."""

_HOVER_CLEARANCE = 0.10
"""Jaw height (m) above the cube when hovering, before/after lowering to grasp."""

_RELEASE_CLEARANCE = 0.02
"""Jaw height (m) above the target marker when lowering to release the cube."""

# Phase boundaries (state-machine step count). `LeRobotRecorderManager` records exactly one
# frame per `env.step()` call regardless of the underlying sim dt / `--step_hz`, so these are
# calibrated as `round(seconds * 30)` to match a 30fps (`--lerobot_dataset_fps 30`) recording.
_APPROACH_STEPS = 60  # phase ends at 60 (2.0s): interpolate from initial EE pos to cube hover
_LOWER_TO_CUBE_END = 90  # +30 (1.0s)
_GRASP_END = 120  # +30 (1.0s)
_LIFT_CUBE_END = 150  # +30 (1.0s)
_MOVE_TO_MIDDLE_END = 195  # +45 (1.5s)
_HOLD_MIDDLE_END = 285  # +90 (3.0s) <- user requirement: hold ~3s near the "middle position"
_MOVE_ABOVE_TARGET_END = 330  # +45 (1.5s)
_LOWER_TO_TARGET_END = 360  # +30 (1.0s)
_RELEASE_END = 380  # +20 (0.67s)
_LIFT_GRIPPER_END = 400  # +20 (0.67s)
_TOTAL_STEPS = 550  # +150 (5.0s): return home, ends the episode


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

    Picks up the cube, holds it for ~3 seconds at the SO-101 calibration "middle position" (all
    joints at 0 rad -- see ``so101_joint_state_server.py``'s "MIDDLE of its range of motion"
    calibration step), then places it on whichever target matches its original spawn side:
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
        self._rest_ee_pos_world: torch.Tensor | None = None
        self._rest_joint_pos: torch.Tensor | None = None
        self._home_start_pos: torch.Tensor | None = None
        self._hold_pos_world: torch.Tensor | None = None
        self._center_x: float = 0.0
        self._target_is_circle: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        """FK calibration for the rest pose and the "middle position" hold target.

        Teleports the joints to two reference configurations in turn -- the SO-101 rest pose,
        then the calibration "middle position" (all joints at 0 rad) -- and reads the resulting
        world-space gripper/jaw positions via FK after each. These EE-space targets are then
        used as ordinary IK targets during the episode, so the arm reaches them through the same
        smooth, physically-simulated motion as every other phase (a direct joint-space teleport
        would not carry a grasped cube realistically).

        ``center_x`` (the left/right decision threshold) is derived from the cube's *nominal*
        (pre-randomization) spawn point, i.e. ``env.cfg.scene.cube.init_state.pos`` -- the same
        fixed value the env config uses to place the two targets and to seed ``center_x`` in
        ``mdp.cube_placed_on_correct_target``/``cube_pick_place_done``. Reading this from the
        config (rather than the live, possibly-already-randomized ``root_pos_w``) guarantees the
        state machine's target choice always agrees with the success-check's.
        """
        robot = env.scene["robot"]
        joint_names = list(robot.data.joint_names)
        ee_frame = env.scene["ee_frame"]

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
        self._rest_ee_pos_world = robot.data.body_pos_w[:, -1, :].clone()

        # "Middle position": SO-101 calibration neutral pose (all joints at 0 rad). Store the
        # resulting *jaw* position (not the raw gripper IK frame) so the hold phases -- which
        # operate in jaw-space, see the phase methods below -- can target it directly.
        middle_joint_pos = torch.zeros_like(self._rest_joint_pos)
        robot.write_joint_state_to_sim(position=middle_joint_pos, velocity=torch.zeros_like(middle_joint_pos))
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._hold_pos_world = ee_frame.data.target_pos_w[:, 1, :].clone()

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
        """Snap to the rest pose before the first recorded step, and blend back to rest at the end.

        The initial snap ensures every recorded episode starts from a consistent rest state
        (rather than whatever pose the previous episode's reset happened to leave the arm in),
        matching the same direct joint-state-write technique used at the end of the episode.
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
        which sits at a fixed but non-trivial offset from the jaw. That offset is read live from
        the ``ee_frame`` sensor (constant here since the commanded gripper orientation never
        changes) and subtracted from every jaw target to get the actual IK command. Using a
        hard-coded gripper-frame offset instead (as tuned for the much larger orange in
        ``pick_orange.py``) placed the gripper's IK point far from this small 3cm cube and the
        jaw never closed around it.
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
        gripper_to_jaw = ee_frame.data.target_pos_w[:, 1, :] - ee_frame.data.target_pos_w[:, 0, :]

        target_quat_w = quat_from_euler_xyz(
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        ).repeat(num_envs, 1)
        target_quat = quat_mul(quat_inv(robot_base_quat_w), target_quat_w)

        if step == 0:
            self._initial_ee_pos = robot.data.body_pos_w[:, -1, :].clone()
            self._target_is_circle = cube_pos_w[:, 0] > self._center_x

        selected_target_pos_w = torch.where(self._target_is_circle.unsqueeze(-1), circle_target_pos_w, target_pos_w)

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
        self._home_start_pos = None
        self._target_is_circle = None

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
