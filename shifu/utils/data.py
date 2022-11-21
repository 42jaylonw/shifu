from shifu.gym.env import ShifuVecEnv
from shifu.configs import BaseEnvConfig
from shifu.utils.torch_utils import free_tensor_attrs
from torch.utils.data import Dataset

import torch
import numpy as np


def to_np(tensor, dtype=np.float32):
    return tensor.detach().cpu().numpy().astype(dtype)


class ShifuDataset(Dataset):
    env: ShifuVecEnv

    def __init__(
            self,
            env_class,
            env_cfg: BaseEnvConfig,
            batch_size,
            num_data,
            render_mode=0,
    ):
        self.num_data = num_data
        env_cfg.debug.headless = not render_mode
        env_cfg.num_envs = batch_size
        self.env = env_class(env_cfg)
        self.env.reset()
        # random starting point
        self.env.episode_length_buf = torch.randint(int(self.env.max_episode_length),
                                                    size=(self.env.num_envs,),
                                                    device=self.env.device)
        self._ctr = 0

    def reset(self):
        self._ctr = 0
        self.env.reset()

    def destroy(self):
        self.env.destroy()
        free_tensor_attrs(self)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if idx >= self.__len__():
            self.destroy()
            raise IndexError("list index out of range")

        actions = torch.rand(self.env.num_envs, self.env.cfg.num_actions, device=self.env.device)
        obs, privileged_obs, rews, dones, infos = self.env.step(actions.detach())

        self._ctr += 1
        return obs, privileged_obs
