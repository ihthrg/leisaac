"""Offline harness: run LiftCubePickPlaceStateMachine against a synthetic SO-101.

Isaac Sim is not needed here. The modules it would provide are stubbed and the robot is replaced
by an analytic forward-kinematics model, which is enough to exercise everything the joint-space
planner does: the nine calibration probes, the closed-form IK, the phase schedule, joint limits,
and the ordering of the emitted action vector.

Because the arm is synthetic, a pass means the *maths* is right, not that the task succeeds in
Isaac Sim -- run ``generate.py`` for that. Its value is that it catches the sign, ordering and
limit mistakes that otherwise only show up after a full simulation run.

Run with::

    python scripts/datagen/state_machine/offline_kinematics_check.py
"""

import math
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
SM_DIR = REPO / "source" / "leisaac" / "leisaac" / "datagen" / "state_machine"


# --------------------------------------------------------------------------------------
# isaaclab / leisaac stubs
# --------------------------------------------------------------------------------------
def _quat_inv(q):
    out = q.clone()
    out[..., 1:] *= -1.0
    return out


def _quat_apply(q, v):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    u = torch.stack([x, y, z], dim=-1)
    return (
        v * (w.unsqueeze(-1) ** 2 - (u * u).sum(-1, keepdim=True))
        + 2.0 * u * (u * v).sum(-1, keepdim=True)
        + 2.0 * w.unsqueeze(-1) * torch.cross(u, v, dim=-1)
    )


def _install_stubs():
    isaaclab = types.ModuleType("isaaclab")
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = lambda name: name
    utils = types.ModuleType("isaaclab.utils")
    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.quat_apply = _quat_apply
    math_mod.quat_inv = _quat_inv

    lerobot = types.ModuleType("leisaac.assets.robots.lerobot")
    lerobot.SO101_FOLLOWER_USD_JOINT_LIMLITS = {
        "shoulder_pan": (-110.0, 110.0),
        "shoulder_lift": (-100.0, 100.0),
        "elbow_flex": (-100.0, 90.0),
        "wrist_flex": (-95.0, 95.0),
        "wrist_roll": (-160.0, 160.0),
        "gripper": (-10.0, 100.0),
    }
    mdp = types.ModuleType("leisaac.tasks.lift_cube.mdp")
    mdp.cube_placed_on_correct_target = lambda *a, **k: torch.ones(1, dtype=torch.bool)

    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.managers": managers,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math_mod,
        "leisaac": types.ModuleType("leisaac"),
        "leisaac.assets": types.ModuleType("leisaac.assets"),
        "leisaac.assets.robots": types.ModuleType("leisaac.assets.robots"),
        "leisaac.assets.robots.lerobot": lerobot,
        "leisaac.tasks": types.ModuleType("leisaac.tasks"),
        "leisaac.tasks.lift_cube": types.ModuleType("leisaac.tasks.lift_cube"),
        "leisaac.tasks.lift_cube.mdp": mdp,
    }.items():
        sys.modules[name] = module


def _load_state_machine():
    import importlib.util

    package = types.ModuleType("sm_pkg")
    package.__path__ = [str(SM_DIR)]
    sys.modules["sm_pkg"] = package

    base = types.ModuleType("sm_pkg.base")

    class StateMachineBase:
        pass

    base.StateMachineBase = StateMachineBase
    sys.modules["sm_pkg.base"] = base

    for name in ("planar_arm_model", "lift_cube_pick_place"):
        spec = importlib.util.spec_from_file_location(f"sm_pkg.{name}", SM_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"sm_pkg.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules["sm_pkg.lift_cube_pick_place"], sys.modules["sm_pkg.planar_arm_model"]


# --------------------------------------------------------------------------------------
# Synthetic robot + environment
# --------------------------------------------------------------------------------------
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ROOT_POS = (0.35, -0.64, 0.01)
ROOT_QUAT = (0.0, 0.0, 0.0, 1.0)  # 180 deg about Z, matching the asset config
FINGER_SWING_PER_RAD = 0.9
"""How far the reported jaw frame rotates about the wrist per radian of gripper opening.

Only one SO-101 finger moves, so the frame Isaac Sim reports swings a long way -- about 8 cm
between the closed and open commands in the real run. This reproduces that, which is what makes
the harness able to catch aiming at the stationary finger instead of the middle of the jaws.
"""


class Data:
    pass


_CLOSED_FOR_GAP = [0.0]
"""Gripper angle the jaws are considered shut at; filled in from the state machine at run time."""

STATIONARY_FINGER_BACKOFF = 0.005
"""How far behind the closed fingertip the stationary finger's contact face actually sits.

Isaac Sim reports the moving finger only, so this offset is invisible to the state machine -- and
it is exactly what broke commanding a jaw width: aiming for a 28 mm grip on the 30 mm cube left a
33 mm gap, so the cube was nudged across and never pinched. Modelling it here means the harness
fails any attempt to go back to computing the width instead of closing until something pushes
back.
"""


class FakeRobot:
    """Analytic SO-101 stand-in that also reproduces the actuator pulling on a written pose.

    ``write_joint_state_to_sim`` sets the joint state, but the physics step that follows drives the
    joints towards the last *position target*. If the caller never writes that target it is still
    all zeros, so the arm slides back out of the pose it was just placed in -- the bug that made
    the real calibration measure a joint gain of 0.806 instead of 1.0. ``pullback`` is the largest
    displacement one step can produce, i.e. the velocity limit times the physics timestep.
    """

    def __init__(self, truth, pullback=0.0):
        self.truth = truth
        self.pullback = pullback
        self.data = Data()
        self.data.joint_names = list(JOINT_NAMES)
        self.data.joint_pos = torch.zeros(1, len(JOINT_NAMES), dtype=torch.float64)
        self.data.root_pos_w = torch.tensor([ROOT_POS], dtype=torch.float64)
        self.data.root_quat_w = torch.tensor([ROOT_QUAT], dtype=torch.float64)
        self.damping_history = []
        self._target = None

    def write_joint_state_to_sim(self, position, velocity=None):
        self.data.joint_pos = position.clone()

    def set_joint_position_target(self, target):
        self._target = target.clone()

    def write_data_to_sim(self):
        pass

    def write_joint_damping_to_sim(self, damping):
        self.damping_history.append(damping)

    def apply_physics_step(self):
        if self.pullback <= 0.0:
            return
        target = self._target if self._target is not None else torch.zeros_like(self.data.joint_pos)
        self.data.joint_pos = self.data.joint_pos + torch.clamp(
            target - self.data.joint_pos, -self.pullback, self.pullback
        )

    def jaw_world(self, gripper=None):
        """World position of the reported jaw frame, which moves with the gripper opening."""
        pan, lift, elbow, wrist_flex = (float(self.data.joint_pos[0, i]) for i in range(4))
        if gripper is None:
            gripper = float(self.data.joint_pos[0, 5])
        pitch1, pitch2, pitch3 = self.truth.pitches_from_joints(lift, elbow, wrist_flex)
        length1, length2, length3 = self.truth.lengths
        wrist_f = self.truth.shoulder[0] + length1 * math.cos(pitch1) + length2 * math.cos(pitch2)
        wrist_h = self.truth.shoulder[1] + length1 * math.sin(pitch1) + length2 * math.sin(pitch2)
        tip_pitch = pitch3 + FINGER_SWING_PER_RAD * gripper
        forward = wrist_f + length3 * math.cos(tip_pitch)
        height = wrist_h + length3 * math.sin(tip_pitch)

        ux, uy = self.truth.plane_dir
        px, py = -uy, ux
        hx = self.truth.pan_axis[0] + forward * ux + self.truth.plane_offset * px
        hy = self.truth.pan_axis[1] + forward * uy + self.truth.plane_offset * py
        angle = self.truth.pan_sign * pan
        bx = math.cos(angle) * hx - math.sin(angle) * hy
        by = math.sin(angle) * hx + math.cos(angle) * hy
        base = torch.tensor([[bx, by, height]], dtype=torch.float64)
        return _quat_apply(self.data.root_quat_w, base) + self.data.root_pos_w

    def grasp_center_world(self, closed, open_, fraction):
        """Point ``fraction`` of the way from the stationary finger to the moving one."""
        shut = self.jaw_world(closed)
        return shut + fraction * (self.jaw_world(open_) - shut)

    def jaw_gap(self, gripper):
        """Distance between the fingertips at the given gripper angle."""
        return float(torch.linalg.vector_norm(self.jaw_world(gripper) - self.jaw_world(_CLOSED_FOR_GAP[0])))

    def blocking_gripper(self, obstacle_width):
        """Gripper angle at which an object of that width stops the jaws.

        ``jaw_gap`` rises monotonically with the opening, so a bisection inverts it exactly, which
        keeps the harness independent of the arc geometry ``jaw_world`` happens to use.
        """
        travel = obstacle_width - STATIONARY_FINGER_BACKOFF
        low, high = _CLOSED_FOR_GAP[0], 2.0
        for _ in range(80):
            mid = 0.5 * (low + high)
            if self.jaw_gap(mid) < travel:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)


class FakeFrame:
    def __init__(self, robot):
        self.robot = robot
        self.data = Data()

    def refresh(self):
        jaw = self.robot.jaw_world()
        self.data.target_pos_w = torch.stack([jaw, jaw], dim=1)


class FakeBody:
    def __init__(self, pos, yaw=0.0):
        self.data = Data()
        self.data.root_pos_w = torch.tensor([pos], dtype=torch.float64)
        self.data.root_quat_w = torch.tensor(
            [[math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]], dtype=torch.float64
        )


class FakeScene(dict):
    def update(self, dt=None):
        self["ee_frame"].refresh()


class FakeEnv:
    def __init__(self, truth, cube_pos, nominal_cube_pos, pullback=0.0):
        self.num_envs = 1
        self.device = "cpu"
        self.physics_dt = 1.0 / 60.0
        robot = FakeRobot(truth, pullback=pullback)
        self.scene = FakeScene(
            robot=robot,
            ee_frame=FakeFrame(robot),
            cube=FakeBody(cube_pos),
            target=FakeBody((nominal_cube_pos[0] + 0.18, nominal_cube_pos[1], nominal_cube_pos[2] - 0.018)),
            circle_target=FakeBody((nominal_cube_pos[0] - 0.18, nominal_cube_pos[1], nominal_cube_pos[2] - 0.018)),
        )
        self.scene.update()
        self.sim = types.SimpleNamespace(step=lambda render=False: (robot.apply_physics_step(), self.scene.update()))
        self.cfg = types.SimpleNamespace(
            scene=types.SimpleNamespace(
                cube=types.SimpleNamespace(init_state=types.SimpleNamespace(pos=nominal_cube_pos))
            )
        )


# --------------------------------------------------------------------------------------
def run(cube_pos, truth_lengths, truth_offsets_deg, truth_signs, label):
    lcpp, pam = MODULES
    truth = pam.PlanarArmModel(
        pan_axis=(0.0, 0.0),
        pan_sign=-1.0,
        plane_dir=(0.0, -1.0),
        plane_offset=0.0196,
        shoulder=(0.0547, 0.1500),
        lengths=truth_lengths,
        pitch_offsets=tuple(math.radians(v) for v in truth_offsets_deg),
        pitch_signs=truth_signs,
    )
    nominal = (0.3460526111597802, -0.312, 0.056)
    # 10 rad/s velocity limit at a 60 Hz timestep -- the same order as the pull that broke the
    # real calibration, so every run here exercises the pose-holding fix.
    env = FakeEnv(truth, cube_pos, nominal, pullback=10.0 / 60.0)
    machine = lcpp.LiftCubePickPlaceStateMachine()
    machine.setup(env)
    robot = env.scene["robot"]
    _CLOSED_FOR_GAP[0] = lcpp._GRIPPER_CLOSE_RAD
    blocking = robot.blocking_gripper(lcpp._CUBE_SIZE)
    limits = machine._joint_limits
    grasp_error = None
    release_error = None
    hold_pitches = None
    hover_tilt = 0.0
    grasp_tilt = 0.0
    max_jump = 0.0
    max_gripper_jump = 0.0
    grip_gap = 0.0
    grasp_end_command = 0.0
    first_gripper = None
    last_gripper = None
    previous = None
    previous_gripper = None
    grasp_target = torch.tensor(cube_pos, dtype=torch.float64) - torch.tensor(
        [0.0, 0.0, lcpp._GRASP_DEPTH_BELOW_CENTER], dtype=torch.float64
    )
    release_target = env.scene["target" if cube_pos[0] <= nominal[0] else "circle_target"].data.root_pos_w[
        0
    ] + torch.tensor([0.0, 0.0, lcpp._RELEASE_CLEARANCE], dtype=torch.float64)

    def grasp_center():
        return robot.grasp_center_world(lcpp._GRIPPER_CLOSE_RAD, lcpp._GRIPPER_OPEN_RAD, machine._model.grasp_fraction)

    while not machine.is_episode_done:
        machine.pre_step(env)
        action = machine.get_action(env)
        step = machine.step_count
        assert action.shape == (1, 6), action.shape
        assert torch.all(action[:, :5] >= limits[:, 0] - 1e-9) and torch.all(action[:, :5] <= limits[:, 1] + 1e-9)
        if previous is not None:
            max_jump = max(max_jump, float((action[0, :5] - previous).abs().max()))
        previous = action[0, :5].clone()
        if previous_gripper is not None:
            max_gripper_jump = max(max_gripper_jump, abs(float(action[0, 5]) - previous_gripper))
        previous_gripper = float(action[0, 5])
        if first_gripper is None:
            first_gripper = previous_gripper
        last_gripper = previous_gripper

        # Perfect tracking, except that the cube physically stops the jaws: below the blocking
        # angle the joint simply cannot follow the command, which is the only cue the state
        # machine gets that it has hold of anything.
        tracked = action.clone()
        if lcpp._LOWER_TO_CUBE_END <= step < lcpp._RELEASE_END:
            tracked[0, 5] = max(float(tracked[0, 5]), blocking)
        robot.write_joint_state_to_sim(tracked)
        env.scene.update()

        angles = {name: float(action[0, i]) for i, name in enumerate(machine._arm_names)}
        pitches = [
            math.degrees(pam.wrap_to_pi(v))
            for v in machine._model.pitches_from_joints(
                angles["shoulder_lift"], angles["elbow_flex"], angles["wrist_flex"]
            )
        ]
        if step == lcpp._APPROACH_STEPS - 1:
            hover_tilt = abs(pitches[2] + 90.0)
        if step == lcpp._GRASP_END - 1:
            grasp_tilt = abs(pitches[2] + 90.0)
            grasp_error = float(torch.linalg.vector_norm(grasp_center()[0] - grasp_target))
            grasp_end_command = float(action[0, 5])
            grip_gap = robot.jaw_gap(float(action[0, 5]))
        if step == lcpp._RELEASE_END - 1:
            release_error = float(torch.linalg.vector_norm(grasp_center()[0] - release_target))
        if step == lcpp._HOLD_MIDDLE_END - 1:
            hold_pitches = pitches
        machine.advance()

    def reachable(point):
        plane = machine._model.to_plane(
            tuple(float(v) for v in _quat_apply(_quat_inv(robot.data.root_quat_w), point - robot.data.root_pos_w)[0])
        )
        span = math.dist(plane, machine._model.shoulder)
        return span <= sum(machine._model.lengths)

    release_reachable = reachable(release_target.unsqueeze(0))
    blocked_gap = robot.jaw_gap(blocking)
    squeeze = blocked_gap - grip_gap
    print(
        f"{label}: grasp centre error = {grasp_error * 1000:.4f} mm | release centre error = "
        f"{release_error * 1000:.4f} mm ({'in' if release_reachable else 'OUT OF'} reach) | "
        f"jaw span = {machine._model.grasp_span * 1000:.1f} mm, grasp at {machine._model.grasp_fraction:.2f} "
        f"of it | jaws stopped at {math.degrees(blocking):.1f} deg, holding "
        f"{math.degrees(machine._grip_hold):.1f} deg = {squeeze * 1000:.1f} mm of squeeze | "
        f"gripper first/last = {math.degrees(first_gripper):.1f}/{math.degrees(last_gripper):.1f} deg | "
        f"hold link pitches = {[round(v, 3) for v in hold_pitches]} deg | "
        f"gripper tilt off vertical: hover {hover_tilt:.2f} deg, grasp {grasp_tilt:.2f} deg | "
        f"max per-step change: arm {math.degrees(max_jump):.2f} deg, gripper "
        f"{math.degrees(max_gripper_jump):.2f} deg"
    )
    assert grasp_error < 1.0e-9, grasp_error
    if release_reachable:
        assert release_error < 1.0e-9, release_error
    # The jaws must notice the cube and then bear on it, and must have finished settling onto it
    # before the phase that lifts it begins.
    assert machine._grip_hold is not None, "the state machine never noticed the cube"
    expected_hold = max(blocking - lcpp._GRIP_SQUEEZE_RAD, lcpp._GRIPPER_CLOSE_RAD)
    assert abs(machine._grip_hold - expected_hold) < 1.0e-9, machine._grip_hold
    assert abs(grasp_end_command - machine._grip_hold) < 1.0e-6, grasp_end_command
    assert squeeze > 0.005, squeeze
    assert abs(first_gripper - lcpp._GRIPPER_REST_RAD) < 1.0e-6, first_gripper
    assert abs(last_gripper - lcpp._GRIPPER_REST_RAD) < 1.0e-3, last_gripper
    assert max(abs(a - b) for a, b in zip(hold_pitches, [90.0, 0.0, 0.0])) < 1.0e-3, hold_pitches
    assert math.degrees(max_jump) < 2.5, max_jump
    assert math.degrees(max_gripper_jump) < 3.0, max_gripper_jump


if __name__ == "__main__":
    _install_stubs()
    MODULES = _load_state_machine()
    for cube in [(0.308, -0.312, 0.056), (0.40, -0.30, 0.056), (0.28, -0.34, 0.056)]:
        run(cube, (0.1159, 0.190, 0.100), (73.59, -12.0, 8.0), (-1.0, -1.0, -1.0), f"cube={cube}")
    run((0.308, -0.312, 0.056), (0.1159, 0.150, 0.140), (73.59, 20.0, -30.0), (-1.0, 1.0, -1.0), "mirrored signs")
    print("all offline checks passed")
