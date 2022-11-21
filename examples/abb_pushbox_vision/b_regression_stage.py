from isaacgym.torch_utils import *

from shifu.units import Box, CameraSensor
from shifu.runner import run_module
from shifu.utils.data import ShifuDataset

import argparse
import numpy as np
import torch
import torch.nn as nn

from examples.abb_pushbox_vision.task_config import (
    AbbRobotConfig,
    TableConfig,
    PushBoxConfig,
    GoalBoxConfig,
    PriorStageEnvConfig,
    PushBoxCameraConfig
)

from examples.abb_pushbox_vision.a_prior_stage import (
    LOG_ROOT,
    AbbRobot,
    AbbPushBox,
    RandPosBox,
    GoalBox
)


def min_max(x):
    return (x - x.min()) / (x.max() - x.min())


class VisionAbbPushBox(AbbPushBox):
    def __init__(self, cfg):
        super(AbbPushBox, self).__init__(cfg)
        self.robot = AbbRobot(AbbRobotConfig())
        self.table = Box(TableConfig())
        self.cube = RandPosBox(PushBoxConfig())
        self.goal = GoalBox(GoalBoxConfig())

        self.camera = CameraSensor(PushBoxCameraConfig())

        self.isg_env.create_envs(
            robot=self.robot,
            objects=[self.table, self.cube, self.goal],
            sensors=[self.camera]
        )
        # buffers
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.ee_history = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float)

    def compute_termination(self):
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.time_out_buf


class MultimodalDataset(ShifuDataset):
    env: VisionAbbPushBox

    def __init__(
            self,
            batch_size,
            num_data,
            render_mode=0
    ):
        super(MultimodalDataset, self).__init__(
            env_class=VisionAbbPushBox,
            env_cfg=PriorStageEnvConfig(),
            batch_size=batch_size,
            num_data=num_data,
            render_mode=render_mode)
        self.env.clip_actions = 10.

    def __getitem__(self, idx):
        if idx >= self.__len__():
            self.destroy()
            raise IndexError("list index out of range")

        all_idx = torch.arange(self.env.num_envs, device=self.env.device)
        self.env.isg_env.reset_idx(all_idx, actors=[self.env.cube, self.env.goal, self.env.robot])
        # generate large random actions
        actions = to_torch(np.random.uniform(-10., 10., (self.env.num_envs, self.env.cfg.num_actions)),
                           device=self.env.device)
        obs, _, rews, dones, infos = self.env.step(actions.detach())

        # image info
        color_tensor = self.env.camera.color_buf.permute(0, 3, 1, 2)
        depth_tensor = self.env.camera.depth_buf.unsqueeze(1)

        data = {
            'rgb': color_tensor,
            'depth': depth_tensor
        }

        labels = {
            'obj_pos': self.env.cube.base_pose[:, :2].detach(),
            'goal_pos': self.env.goal.base_pose[:, :2].detach(),
            'ee_pos': self.env.robot.ee_pose[:, 0, :2].detach(),
        }

        self._ctr += 1
        return data, labels


def get_multi_regressor():
    from shifu.models.autoencoders import (
        ConvEncoder,
        MultimodalAE,
        Decoder
    )
    m_latent = 32
    activation = nn.ReLU(True)
    rgb_encoder = ConvEncoder(in_channels=3, latent_dim=m_latent, activation=activation)
    depth_encoder = ConvEncoder(in_channels=1, latent_dim=m_latent, activation=activation)

    obj_pos_regressor = Decoder(input_dim=m_latent, output_dim=2, hidden_dims=[16, 8])
    goal_pos_decoder = Decoder(input_dim=m_latent, output_dim=2, hidden_dims=[16, 8])
    ee_pos_regressor = Decoder(input_dim=m_latent, output_dim=2, hidden_dims=[16, 8])

    mae = MultimodalAE(
        encoders={'rgb': rgb_encoder, 'depth': depth_encoder},
        decoders={'obj_pos': obj_pos_regressor, 'goal_pos': goal_pos_decoder, 'ee_pos': ee_pos_regressor},
        latent_dim=m_latent
    )
    return mae


def get_args():
    parser = argparse.ArgumentParser("Abb Robot push box task")
    parser.add_argument("--run-mode",
                        '-r',
                        type=str,
                        choices=['train', 'play'],
                        help="run mode for push box task")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    run_args = get_args()
    regressor = get_multi_regressor()
    ds = MultimodalDataset(batch_size=128, num_data=100_000)
    run_module(run_args.run_mode,
               model=regressor,
               dataset=ds,
               model_name='MultiAE-l32',
               log_root=f"{LOG_ROOT}/Regression")
