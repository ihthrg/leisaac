# State Machine Data Generation

The state machine module provides automated data collection for manipulation tasks without human teleoperation. It runs a scripted policy and records demonstrations to HDF5 datasets.

## Recording

```shell
python scripts/datagen/state_machine/generate.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --num_demos 50
```

<details>
<summary><strong>Parameter descriptions for generate.py</strong></summary>

- `--task`: Task environment name to run, e.g., `LeIsaac-SO101-PickOrange-v0`. See [here](/resources/available_env) for available tasks.

- `--num_envs`: Number of parallel simulation environments. Every environment runs the same scripted timeline against its own randomised scene, and is judged and recorded on its own, so this multiplies the demonstrations collected per wall-clock second. Use a count that tiles square — `1`, `4`, `9`, `16` — because the others cost recorded image detail; see [Checking recorded image quality](#checking-recorded-image-quality). LeRobot recording buffers one episode per environment in memory (roughly 1.3 GB each at 640x480 with two cameras), so raise it gradually.

- `--device`: Computation device, such as `cpu` or `cuda` (GPU).

- `--enable_cameras`: Enable camera sensors to collect visual data.

- `--seed`: Seed for the environment. Defaults to current timestamp if not set.

- `--record`: Enable data recording; saves demonstrations to an HDF5 file.

- `--dataset_file`: Path to save the recorded dataset, e.g., `./datasets/pick_orange.hdf5`.

- `--resume`: Resume recording from an existing dataset file.

- `--num_demos`: Number of successful demonstrations to record. Set to `0` for infinite.

- `--step_hz`: Environment stepping rate in Hz (default: `60`).

- `--quality`: Enable quality render mode.

- `--use_lerobot_recorder`: Record directly in LeRobot Dataset format instead of HDF5.

- `--lerobot_dataset_repo_id`: HuggingFace dataset repository ID (format: `username/repository_name`). Required when `--use_lerobot_recorder` is set.

- `--lerobot_dataset_fps`: Dataset frame rate when using LeRobot recorder (default: `30`).

</details>

::::tip
Grasp success rate depends heavily on orange spawn positions. Adjusting the spawn positions in the task's environment config file (e.g. moving oranges closer to the robot base) can significantly improve success rate.
::::

## Checking recorded image quality

Isaac Lab renders every environment's camera into one shared buffer, laid out as `ceil(sqrt(n))` columns by however many rows that needs. The buffer therefore has the cameras' aspect ratio multiplied by `columns / rows`, and when those are not equal the recorded images lose detail along one axis. The dataset still looks perfectly well-formed — the right frame count, contiguous timestamps, video that plays — so nothing downstream complains.

Only counts that tile square are safe. Measured on this task with `image_sharpness_check.py`, where the number is the vertical detail remaining relative to the horizontal:

| `--num_envs` | grid | buffer | detail ratio |
| --- | --- | --- | --- |
| 1 | 1x1 | 640x480 | 0.70 |
| 2 | 2x1 | 1280x480 | 0.41 |
| 4 | 2x2 | 1280x960 | 0.70 |

So `--num_envs 2` is the configuration to avoid, and `--num_envs 4` collects four times as fast at full quality. `--quality` does not recover the loss; it is not an anti-aliasing problem. `1, 4, 9, 16, 25` and so on all tile square, and `generate.py` warns when the count you asked for does not.

To check a recording yourself, record the same task twice changing only `--num_envs`, then compare them:

```shell
python scripts/datagen/state_machine/image_sharpness_check.py \
    ./datasets/check_1env.hdf5 ./datasets/check_2env.hdf5
```

The script needs neither Isaac Sim nor this package, and reads LeRobot mp4s as well as HDF5 recordings. It reports the detail left along each image axis and says which of the two plausible culprits fits: an even loss on both axes is video compression, while a loss concentrated on one axis means the frames were already smeared when they reached the recorder.

## Replay

After recording, you can replay the collected demonstrations in simulation:

```shell
python scripts/datagen/state_machine/replay.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --task_type so101_state_machine \
    --select_episodes 0 \
    --device cuda \
    --enable_cameras \
    --replay_mode action
```

<details>
<summary><strong>Parameter descriptions for replay.py</strong></summary>

- `--task`: Task environment name to run, e.g., `LeIsaac-SO101-PickOrange-v0`.

- `--num_envs`: Number of parallel simulation environments, usually `1`.

- `--device`: Computation device, such as `cpu` or `cuda` (GPU).

- `--enable_cameras`: Enable camera sensors to visualize during replay.

- `--dataset_file`: Path to the recorded dataset, e.g., `./datasets/pick_orange.hdf5`.

- `--replay_mode`: Replay mode — `action` replays IK pose targets, `state` replays joint positions.

- `--task_type`: State machine device type used during recording, e.g., `so101_state_machine` or `bi_so101_state_machine`. Inferred from task name if not set.

- `--select_episodes`: List of episode indices to replay. Leave empty to replay all episodes.

- `--step_hz`: Environment stepping rate in Hz (default: `60`).

</details>

## Adding a New Task

1. Implement a `StateMachineBase` subclass in `source/leisaac/leisaac/datagen/state_machine/`.
2. Register it in `TASK_REGISTRY` inside `scripts/datagen/state_machine/generate.py`:

```python
TASK_REGISTRY = {
    "LeIsaac-SO101-PickOrange-v0": (PickOrangeStateMachine, "so101_state_machine"),
    "LeIsaac-MY-NewTask-v0":       (MyNewStateMachine,      "so101_state_machine"),
}
```
