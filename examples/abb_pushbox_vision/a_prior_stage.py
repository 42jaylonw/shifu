from isaacgym.torch_utils import *
from isaacgym import gymtorch

from shifu.units import ArmRobot, Box
from shifu.gym import ShifuVecEnv
from shifu.runner import run_policy

import argparse
import numpy as np
import torch

from examples.abb_pushbox_vision.task_config import (
    AbbRobotConfig,
    TableConfig,
    PushBoxConfig,
    GoalBoxConfig,
    PriorStageEnvConfig,
    PriorStagePPOConfig
)

LOG_ROOT = './logs/abb_pushbox_vision'


class RandPosBox(Box):
    def __init__(
            self,
            cfg,
    ):
        super().__init__(cfg)
        self.pos_range = {
            "low": [-0.1, -0.1, 0.125],
            "high": [0.1, 0.1, 0.125]
        }
        self.euler_range = {
            "low": [0, 0, -np.pi],
            "high": [0, 0, np.pi]
        }

    def _reset_root_state(self, env_ids):
        object_indices = self.root_indices[env_ids]
        rand_pos = np.array([np.random.uniform(**self.pos_range) for _ in env_ids])
        rand_pos = to_torch(rand_pos, dtype=torch.float, device=self.device)

        rand_euler = to_torch(np.array([np.random.uniform(**self.euler_range) for _ in env_ids]), device=self.device)
        rand_quat = quat_from_euler_xyz(rand_euler[:, 0], rand_euler[:, 1], rand_euler[:, 2])

        self.env.root_state[object_indices, :3] = rand_pos
        self.env.root_state[object_indices, 3:7] = rand_quat
        self.env.root_state[object_indices, 7:13] = 0.

        return object_indices


class GoalBox(RandPosBox):
    def __init__(self, cfg):
        super(GoalBox, self).__init__(cfg)
        self.pos_range['low'][2] = self.cfg.default_pos[2]
        self.pos_range['high'][2] = self.cfg.default_pos[2]


class AbbRobot(ArmRobot):
    def init_buffers(self):
        super(AbbRobot, self).init_buffers()
        self.min_ee_pos = to_torch(self.cfg.min_ee_pos, device=self.device)
        self.max_ee_pos = to_torch(self.cfg.max_ee_pos, device=self.device)

    def step(self, actions):
        tar_pos = self.ee_pose[:, 0, :3] + actions * self.end_effector_velocity * self.env.dt
        tar_pos = torch.clip(tar_pos, self.min_ee_pos, self.max_ee_pos)
        tar_quat = torch.Tensor([0., 1., 0., 0]).repeat((self.env.num_envs, 1)).to(self.device)
        ee_pos_targets = torch.cat([tar_pos, tar_quat], dim=1)
        self.dof_targets[:] = self.inverse_kinematics(ee_pos_targets)
        self.apply_dof_targets(self.dof_targets)


class AbbPushBox(ShifuVecEnv):
    def __init__(self, cfg):
        super(AbbPushBox, self).__init__(cfg)
        self.robot = AbbRobot(AbbRobotConfig())
        self.table = Box(TableConfig())
        self.cube = RandPosBox(PushBoxConfig())
        self.goal = GoalBox(GoalBoxConfig())

        self.isg_env.create_envs(
            robot=self.robot,
            objects=[self.table, self.cube, self.goal],
        )

        # buffers
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def episode_log(self, env_ids):
        return {'success_rate': torch.mean(self.success_buf.to(torch.float)[env_ids])}

    def compute_observations(self):
        self.obs_buf = torch.cat([
            self.cube.base_pose[:, :2],
            self.goal.base_pose[:, :2],
            self.robot.ee_pose[:, 0, :2],
        ], dim=1)

    def compute_termination(self):
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.success_buf = self.is_success().to(torch.bool)
        obj_outbound = (torch.any(self.cube.base_pose[:, :2] < self.robot.min_ee_pos[:2], dim=1) |
                        torch.any(self.cube.base_pose[:, :2] > self.robot.max_ee_pos[:2], dim=1))
        ee_outbound = (torch.any(self.robot.ee_pose[:, 0, :2] < self.robot.min_ee_pos[:2], dim=1) |
                       torch.any(self.robot.ee_pose[:, 0, :2] > self.robot.max_ee_pos[:2], dim=1))
        outbound = obj_outbound | ee_outbound
        self.reset_buf = self.time_out_buf | outbound | self.success_buf

    def build_reward_functions(self):
        return [
            self.reward_reaching,
            self.reward_success
        ]

    def reward_reaching(self):
        curr_dist = torch.linalg.norm(self.goal.base_pose[:, :2] - self.cube.base_pose[:, :2], axis=1)
        ee_obj_dist = torch.linalg.norm(self.robot.ee_pose[:, 0, :2] - self.cube.base_pose[:, :2], axis=1)
        in_ws = (ee_obj_dist < 0.1).to(torch.long)
        reach_rew = in_ws * torch.exp(-torch.square(curr_dist) / 0.05)
        return reach_rew

    def reward_success(self):
        success_rew = self.is_success().to(torch.float)
        return success_rew * 200

    def is_success(self):
        obj_goal_pos_dist = torch.linalg.norm(self.goal.base_pose[:, :2] - self.cube.base_pose[:, :2], axis=1)
        return (obj_goal_pos_dist < 0.02).to(torch.long)  # less than 0.02 [m]

    def in_ws(self):
        ee_obj_dist = torch.linalg.norm(self.robot.ee_pose[:, 0, :3] - self.cube.base_pose[:, :3], axis=1)
        return (ee_obj_dist < 0.1).to(torch.long)


def get_args():
    parser = argparse.ArgumentParser("Abb Robot push box task")
    parser.add_argument("--run-mode",
                        '-r',
                        type=str,
                        choices=['train', 'play', 'random'],
                        help="run mode for push box task")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    run_args = get_args()

    run_policy(
        run_mode=run_args.run_mode,
        env_class=AbbPushBox,
        env_cfg=PriorStageEnvConfig(),
        policy_cfg=PriorStagePPOConfig(),
        log_root=f"{LOG_ROOT}/Prior",
    )
