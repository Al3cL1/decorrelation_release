"""Zarr dataset for the DP baselines: reads the keys in shape_meta from the FDP zarrs and turns
images from HWC uint8 on disk into CHW float in [0, 1].
"""

import copy
from typing import Dict, Optional

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class RobocasaZarrDataset(BaseImageDataset):
    """Read a RoboCasa zarr directly from disk (no copy to RAM)."""

    def __init__(
        self,
        shape_meta: dict,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        action_key: str = "action",
        in_ram: bool = False,
    ):
        self.action_key = action_key
        # parse obs keys by type
        rgb_keys: list = []
        lowdim_keys: list = []
        for key, attr in shape_meta["obs"].items():
            t = attr.get("type", "low_dim")
            if t == "rgb":
                rgb_keys.append(key)
            elif t == "low_dim":
                lowdim_keys.append(key)

        # in_ram: copy into memory first, as FDP's ZarrDataset does (much faster on network filesystems)
        if in_ram:
            replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=[action_key, *rgb_keys, *lowdim_keys],
            )
        else:
            replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

        # only take first n_obs_steps observations per key
        key_first_k: dict = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        if max_train_episodes is not None:
            train_mask = downsample_mask(
                mask=train_mask, max_n=max_train_episodes, seed=seed
            )

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    # ------------------------------------------------------------------
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    # ------------------------------------------------------------------
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action — range normalizer maps observed range → [-1, 1]
        normalizer["action"] = get_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer[self.action_key])
        )

        # obs
        for key in self.lowdim_keys:
            # zarr 3 Arrays don't have .astype; materialize to numpy first.
            data = np.asarray(self.replay_buffer[key], dtype=np.float32)
            stat = array_to_stats(data)
            if key.endswith("quat"):
                # quaternion components are already in [-1, 1]
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                # pos, gripper, task_id — map to [-1, 1]
                normalizer[key] = get_range_normalizer_from_stat(stat)

        # rgb — maps uint8 [0, 255] / 255 → [0, 1], then to [-1, 1] inside policy
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    # ------------------------------------------------------------------
    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key][:].astype(np.float32))

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)  # no-op when n_obs_steps is None

        obs_dict: dict = {}

        # images: (To, H, W, C) uint8 → (To, C, H, W) float32 [0, 1]
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]

        # state: cast to float32 (task_id is int32 on disk)
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data[self.action_key].astype(np.float32)),
        }
