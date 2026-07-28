"""Unified data generation script using state machines.

Selects the appropriate state machine based on --task and runs the recording loop.

Usage:
    python scripts/datagen/state_machine/generate.py \
        --task LeIsaac-SO101-PickOrange-v0 \
        --num_envs 1 --device cuda --enable_cameras \
        --record --dataset_file ./datasets/pick_orange.hdf5 --num_demos 50
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import math
import os
import signal
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="State machine data generation for LeIsaac tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--record", action="store_true", help="Whether to enable record function.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="Whether to resume recording in the existing dataset file.")
parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument("--quality", action="store_true", help="Whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="Whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.datagen.state_machine import (
    LiftCubePickPlaceStateMachine,
    PickOrangeStateMachine,
)
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

# Maps gym task id → (StateMachineClass, device_type)
TASK_REGISTRY = {
    "LeIsaac-SO101-PickOrange-v0": (PickOrangeStateMachine, "so101_state_machine"),
    "LeIsaac-SO101-LiftCubePickPlace-v0": (LiftCubePickPlaceStateMachine, "so101_cube_state_machine"),
}


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def auto_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: torch.Tensor):
    """Mark each environment as succeeded or failed independently.

    ``success`` carries one verdict per environment, so a batch of parallel environments keeps the
    demonstrations that worked instead of discarding all of them because one missed.
    """
    if hasattr(env, "termination_manager"):
        env.termination_manager.set_term_cfg(
            "success",
            TerminationTermCfg(func=lambda env, verdict=success.clone(): verdict),
        )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = bool(success.all().item())


def _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name):
    """Configure termination and recorder settings on env_cfg."""
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.auto_terminate = True
    else:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None


def _replace_recorder_manager(env, env_cfg, args_cli):
    """Replace the default recorder manager with streaming or lerobot recorder."""
    del env.recorder_manager
    if args_cli.use_lerobot_recorder:
        from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
        from leisaac.enhance.managers.lerobot_recorder_manager import (
            LeRobotRecorderManager,
        )

        dataset_cfg = LeRobotDatasetCfg(
            repo_id=args_cli.lerobot_dataset_repo_id,
            fps=args_cli.lerobot_dataset_fps,
        )
        env.recorder_manager = LeRobotRecorderManager(
            env_cfg.recorders,
            dataset_cfg,
            env,
            step_hz=args_cli.step_hz,
        )
    else:
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"


def _on_episode_done(env, sm, args_cli, resume_recorded_demo_count, current_recorded_demo_count, start_record_state):
    """Handle end-of-episode logic. Returns (current_recorded_demo_count, start_record_state, should_break)."""
    try:
        success = sm.check_success(env)
    except Exception as e:
        print("Success check failed:", e)
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    succeeded = int(success.sum().item())
    if env.num_envs == 1:
        print("Episode success!" if succeeded else "Episode failed!")
    else:
        print(f"{succeeded}/{env.num_envs} environments succeeded.")

    if start_record_state:
        if args_cli.record:
            print("Stop Recording!!!")
        start_record_state = False

    if args_cli.record:
        auto_terminate(env, success)
        current_recorded_demo_count += succeeded
    else:
        auto_terminate(env, torch.zeros_like(success))

    if (
        args_cli.record
        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
        > current_recorded_demo_count
    ):
        current_recorded_demo_count = (
            env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
        )
        print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

    if (
        args_cli.record
        and args_cli.num_demos > 0
        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count >= args_cli.num_demos
    ):
        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
        return current_recorded_demo_count, start_record_state, True

    env.reset()
    sm.reset()
    auto_terminate(env, torch.zeros_like(success))

    if args_cli.record and args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
        return current_recorded_demo_count, start_record_state, True

    return current_recorded_demo_count, start_record_state, False


def _tile_grid(num_envs: int) -> tuple[int, int]:
    """Return the (columns, rows) Isaac Lab will use to pack every environment's camera into one buffer.

    Mirrors ``isaaclab_ov/renderers/ovrtx_usd.py::_tiled_resolution``.
    """
    columns = math.ceil(math.sqrt(num_envs))
    return columns, math.ceil(num_envs / columns)


def _warn_about_lopsided_tiles(num_envs: int) -> None:
    """Warn when the tiled render buffer will not have the cameras' own aspect ratio.

    Isaac Lab renders every environment's camera into a single buffer of ``columns * width`` by
    ``rows * height``, so the buffer is the camera's shape multiplied by ``columns / rows``. When
    those are not equal the renderer loses detail along one axis, and nothing downstream says so:
    the images come out the right size and merely look soft.

    Measured with ``image_sharpness_check.py``, the vertical/horizontal detail ratio of a recording
    was 0.70 at ``--num_envs 1`` (a 1x1 grid), fell to 0.41 at ``--num_envs 2`` (2x1, a buffer
    twice as wide as it is tall relative to the camera), and returned to 0.70 at ``--num_envs 4``
    (2x2). ``--quality`` did not recover it. So parallelism is not the problem -- the tile layout
    is, and it costs nothing to pick a count that squares up.
    """
    columns, rows = _tile_grid(num_envs)
    if columns == rows:
        return

    # A lopsided grid always has fewer rows than columns, so the nearest square grids are the ones
    # that fill (columns - 1) and columns square. Both are suggested full, because a full grid
    # costs the same to render as a partial one of the same shape.
    fewer, more = (columns - 1) ** 2, columns**2

    print(
        f"[generate][warning] --num_envs {num_envs} tiles the cameras {columns}x{rows}, so the render"
        f" buffer is {columns}/{rows} times the cameras' aspect ratio. Recorded images lose detail"
        " along one axis when that happens, and they still come out looking like ordinary images."
        f" Consider --num_envs {fewer} or {more} instead, which tile square."
        " Check any recording with scripts/datagen/state_machine/image_sharpness_check.py."
    )


def main():
    """Run a state machine in a LeIsaac manipulation environment."""
    task_name = args_cli.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{task_name}' is not registered in TASK_REGISTRY.\nAvailable tasks: {list(TASK_REGISTRY.keys())}"
        )
    SMClass, device = TASK_REGISTRY[task_name]

    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args_cli.record:
        _warn_about_lopsided_tiles(args_cli.num_envs)

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    if args_cli.quality:
        # Matches teleop_se3_agent.py. This script accepted --quality and then ignored it, so a
        # run asking for better rendering got the default and said nothing about it -- which is
        # the worst way for a flag to fail, because the output still looks like an answer. Note
        # that it does not rescue the detail lost to a lopsided tile grid; see
        # _warn_about_lopsided_tiles for that.
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    is_direct_env = "Direct" in task_name
    _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # disable gravity for every robot link prim
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    if args_cli.record:
        _replace_recorder_manager(env, env_cfg, args_cli)

    rate_limiter = RateLimiter(args_cli.step_hz)

    if hasattr(env, "initialize"):
        env.initialize()

    # one-time state machine setup (e.g. FK calibration)
    sm = SMClass()
    sm.setup(env)
    env.reset()
    sm.reset()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    start_record_state = False
    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        while simulation_app.is_running() and not simulation_app.is_exiting() and not interrupted:
            with torch.inference_mode():
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, device)

                if sm.is_episode_done:
                    current_recorded_demo_count, start_record_state, should_break = _on_episode_done(
                        env, sm, args_cli, resume_recorded_demo_count, current_recorded_demo_count, start_record_state
                    )
                    if should_break:
                        break
                else:
                    if not start_record_state:
                        if args_cli.record:
                            print("Start Recording!!!")
                        start_record_state = True

                    sm.pre_step(env)
                    actions = sm.get_action(env)
                    env.step(actions)
                    sm.advance()

                if rate_limiter:
                    rate_limiter.sleep(env)

            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
