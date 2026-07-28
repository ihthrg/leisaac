from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import DatasetExportMode, RecorderManager
from isaaclab.utils.datasets.episode_data import EpisodeData
from leisaac.utils.robot_utils import build_feature_from_env

from ..datasets.lerobot_dataset_handler import LeRobotDatasetCfg, LeRobotDatasetHandler
from .recorder_manager import EnhanceDatasetExportMode


class LeRobotRecorderManager(RecorderManager):
    """Recorder that writes episodes in the LeRobot dataset format.

    A ``LeRobotDataset`` has exactly one episode buffer: ``add_frame`` appends to it and
    ``save_episode`` commits it. With several environments running at once their frames would
    interleave into a single nonsense episode, so frames are collected per environment here and
    replayed into the dataset one environment at a time when the episode ends.

    That buffering holds one episode's frames -- images included -- in memory for every
    environment. At 640x480 with two cameras and 30 fps that is roughly 1.8 MB per frame, so a
    23 s episode costs about 1.3 GB per environment. Choose ``--num_envs`` with that in mind.
    """

    def __init__(
        self,
        cfg: object,
        dataset_cfg: LeRobotDatasetCfg,
        env: ManagerBasedEnv,
        step_hz: int | None = None,
    ) -> None:
        real_export_mode = cfg.dataset_export_mode
        cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        super().__init__(cfg, env)
        cfg.dataset_export_mode = real_export_mode

        assert cfg.dataset_export_mode in [
            DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
            EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME,
        ], "only support EXPORT_SUCCEEDED_ONLY|EXPORT_SUCCEEDED_ONLY_RESUME"

        cfg.dataset_file_handler_class_type = LeRobotDatasetHandler

        resume = cfg.dataset_export_mode == EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME

        dataset_cfg.robot_type = env.cfg.robot_name
        dataset_cfg.features = build_feature_from_env(env, dataset_cfg)

        self._dataset_cfg = dataset_cfg

        self._dataset_file_handler = cfg.dataset_file_handler_class_type(dataset_cfg)
        self._dataset_file_handler.create(None, resume=resume)

        self._skip_frames = 5
        self._record_every_n_steps = 1
        if step_hz is not None:
            if step_hz < dataset_cfg.fps or step_hz % dataset_cfg.fps != 0:
                raise ValueError(
                    "LeRobot recording requires step_hz to be an integer multiple of dataset fps; "
                    f"got step_hz={step_hz}, dataset_fps={dataset_cfg.fps}."
                )
            self._record_every_n_steps = step_hz // dataset_cfg.fps
        self._env_steps_record = torch.zeros(self._env.num_envs)
        self._pending_frames: dict[int, list[dict]] = {env_id: [] for env_id in range(self._env.num_envs)}

    def __str__(self) -> str:
        msg = "[Enhanced] LeRobotRecorderManager. \n"
        msg += super().__str__()
        return msg

    def finalize(self):
        self._dataset_file_handler.finalize()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        self._env_steps_record[env_ids] = 0
        for env_id in env_ids:
            self._pending_frames[int(env_id)] = []
        return super().reset(env_ids)

    def record_post_step(self) -> None:
        """Buffer this step's frame for every environment."""
        super().record_post_step()

        skipped_steps = self._skip_frames * self._record_every_n_steps
        for env_idx in range(self._env.num_envs):
            self._env_steps_record[env_idx] += 1
            step = int(self._env_steps_record[env_idx].item())
            if step <= skipped_steps:
                continue
            if (step - skipped_steps - 1) % self._record_every_n_steps != 0:
                self._episodes[env_idx]._data.clear()
                continue
            frame = self._env.cfg.build_lerobot_frame(self._episodes[env_idx], self._dataset_cfg)
            self._pending_frames[env_idx].append(frame)
            self._episodes[env_idx]._data.clear()

    def export_episodes(self, env_ids: Sequence[int] | None = None) -> None:
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # Export episode data through dataset exporter, one environment at a time: the dataset
        # holds a single episode buffer, so these cannot be interleaved.
        for env_id in env_ids:
            if env_id in self._episodes:
                episode_succeeded = self._episodes[env_id].success
                target_dataset_file_handler = self._dataset_file_handler
                if episode_succeeded:
                    for frame in self._pending_frames[env_id]:
                        target_dataset_file_handler.add_frame(frame)
                    target_dataset_file_handler.flush()
                    self._exported_successful_episode_count[env_id] = (
                        self._exported_successful_episode_count.get(env_id, 0) + 1
                    )
                else:
                    target_dataset_file_handler.clear()
                    self._exported_failed_episode_count[env_id] = (
                        self._exported_failed_episode_count.get(env_id, -1) + 1
                    )  # default to -1 to handle the first reset
                self._pending_frames[env_id] = []
            # Reset the episode buffer for the given environment after export
            self._episodes[env_id] = EpisodeData()

    def record_pre_reset(self, env_ids: Sequence[int] | None, force_export_or_skip=None) -> None:
        """
        Modified from super().record_pre_reset() with additional logic to retrieve success status values from _get_dones()
        to adapt to RecorderEnhanceDirectRLEnv.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        for term in self._terms.values():
            key, value = term.record_pre_reset(env_ids)
            self.add_to_episodes(key, value, env_ids)

        # Set task success values for the relevant episodes
        success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
        # Check success indicator from termination terms
        if hasattr(self._env, "termination_manager"):  # for ManagerBasedEnv
            if "success" in self._env.termination_manager.active_terms:
                success_results |= self._env.termination_manager.get_term("success")[env_ids]
        elif hasattr(self._env, "_get_dones"):  # for DriectEnv
            done, _ = self._env._get_dones()
            success_results |= done[env_ids]
        self.set_success_to_episodes(env_ids, success_results)

        if force_export_or_skip or (force_export_or_skip is None and self.cfg.export_in_record_pre_reset):
            self.export_episodes(env_ids)
