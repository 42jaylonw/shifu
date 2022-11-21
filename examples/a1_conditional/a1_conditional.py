import typing

from isaacgym import gymapi
from isaacgym.torch_utils import *

from shifu.units import LeggedRobot
from shifu.gym import ShifuVecEnv
from shifu.runner import run_policy

import torch
import numpy as np

from examples.a1_conditional.task_config import (
    A1ActorConfig,
    A1EnvConfig,
    A1PPOConfig
)

LOG_ROOT = "./logs/a1_conditional"


class A1Robot(LeggedRobot):
    def __init__(self, cfg):
        super(A1Robot, self).__init__(cfg)
        self.p_gains = to_torch(self.cfg.dof_stiffness)
        self.d_gains = to_torch(self.cfg.dof_damping)

    def random_rigid_shape_props(self, env_ids, rigid_shape_props):
        for prop in rigid_shape_props:
            prop.friction = np.random.uniform(0.5, 1.25)
        return rigid_shape_props

    def load_to(self, env_id, env_handle, seg_id):
        super(A1Robot, self).load_to(env_id, env_handle, seg_id)

        # set legs colors
        orange = gymapi.Vec3(1., 0.3412, 0.1)
        for name, idx in self.rigid_body_dict.items():
            if 'hip' in name or 'thigh' in name or 'calf' in name:
                self.gym.set_rigid_body_color(
                    env_handle, self.actor_handle, idx, gymapi.MESH_VISUAL_AND_COLLISION, orange)

    def _reset_root_state(self, env_ids):
        robot_indices = self.root_indices[env_ids]
        self.env.root_state[robot_indices, :3] = self.default_base_pose[:3] + self.env.env_origins[env_ids]
        # set rand x y
        rand_xy = torch_rand_float(-1, 1, shape=(len(env_ids), 2), device=self.device)
        self.env.root_state[robot_indices, :2] += rand_xy
        self.env.root_state[robot_indices, 3:7] = self.default_base_pose[3:7]
        self.env.root_state[robot_indices, 7:] = 0.  # [7:10]: lin vel, [10:13]: ang vel

    def init_buffers(self):
        super(A1Robot, self).init_buffers()
        self.torques = torch.zeros(self.env.num_envs,
                                   self.num_dof,
                                   dtype=torch.float,
                                   device=self.device,
                                   requires_grad=False)
        leg_indices = [self.rigid_body_dict[n] for n in self.rigid_body_dict.keys()
                       if "thigh" in n or "calf" in n]
        self.leg_indices = to_torch(leg_indices, dtype=torch.long, device=self.device)
        self.rand_force_buf = torch.zeros(self.env.num_envs, self.num_bodies, 3, device=self.device)

    def step(self, actions):
        for _ in range(self.env.decimation):
            self.torques = self.p_gains * (actions + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
            self.torques = torch.clip(self.torques, -self.torque_limits, self.torque_limits)
            self._internal_motor_step(self.torques)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_step()

        self.apply_force_on_base(self.rand_force_buf.view(-1, 3))

    def reset_idx(self, env_ids):
        super(A1Robot, self).reset_idx(env_ids)
        # update
        self.update_rand_force_buf(env_ids)

    def update_rand_force_buf(self, env_ids):
        max_force = 5.
        # x, y, z
        base_id = self.rigid_body_dict['base']
        self.rand_force_buf[env_ids, base_id] = torch_rand_float(
            -max_force, max_force, (len(env_ids), 3), device=self.device)


class A1Conditional(ShifuVecEnv):
    def __init__(self, cfg):
        super(A1Conditional, self).__init__(cfg)
        self.robot = A1Robot(A1ActorConfig())
        self.isg_env.create_envs(robot=self.robot)
        self._init_command()

        # buffer
        self.contact_terminate_indices = self.isg_env.gym.find_actor_rigid_body_handle(
            self.isg_env.env_handles[0], self.robot.actor_handle, 'base')
        self.swing_time = torch.zeros(self.num_envs, self.robot.ee_indices.shape[0], dtype=torch.float,
                                      device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.robot.ee_indices), dtype=torch.bool,
                                         device=self.device, requires_grad=False)

        self.contact_terminate_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _init_command(self):
        # buffer
        self.cmd_lin_vel_x = [-1., 1.]  # [m/s]
        self.cmd_lin_vel_y = [-1., 1.]  # [m/s]
        self.cmd_ang_vel_yaw = [-1., 1.]  # [rad/s]
        self.num_commands = 3
        self.command_buf = torch.zeros(self.num_envs, self.num_commands, dtype=torch.float32, device=self.device)

    def reset_idx(self, env_ids):
        if self.cfg.terrain.curriculum:
            self.update_terrain_curriculum(env_ids)
        super(A1Conditional, self).reset_idx(env_ids)
        self.sample_command(env_ids)

    def step(self, actions: torch.Tensor):
        scaled_action = actions * 0.5
        return super(A1Conditional, self).step(scaled_action)

    def episode_log(self, env_ids) -> typing.Dict:
        return {
            "terrain_levels": torch.mean(self.terrain_levels.to(torch.float)),
        }

    def compute_observations(self):
        heights = torch.clip(self.robot.base_pose[:, 2].unsqueeze(1) - 0.5 -
                             self.isg_env.measured_heights, -1, 1.)

        self.obs_buf = torch.cat([
            self.command_buf,
            self.robot.base_lin_vel,
            self.robot.base_ang_vel,
            self.robot.gravity_vec,
            self.robot.dof_pos - self.robot.default_dof_pos,
            self.robot.dof_vel,
            self.actions_recorder.flatten(),
            heights
        ], dim=1)

    def compute_termination(self):
        self.contact_terminate_buf = torch.norm(
            self.robot.contact_forces[:, self.contact_terminate_indices, :], dim=-1) > 1.
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = self.time_out_buf | self.contact_terminate_buf

    def build_reward_functions(self) -> typing.List:
        return [
            self.tracking_lin_vel,
            self.tracking_ang_vel,
            self.stabilizing_base,
            self.smoothing_action,
            self.leg_collision,
            self.torques_penalize
        ]

    def tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.command_buf[:, :2] - self.robot.base_lin_vel[:, :2]), dim=1)
        return 1.0 * torch.exp(-lin_vel_error / 0.25)

    def tracking_ang_vel(self):
        ang_vel_error = torch.square(self.command_buf[:, 2] - self.robot.base_ang_vel[:, 2])
        return 0.5 * torch.exp(-ang_vel_error / 0.25)

    def stabilizing_base(self):
        z_vel = -2.0 * torch.square(self.robot.base_lin_vel[:, 2])
        ang_vel = -0.005 * torch.sum(torch.square(self.robot.base_ang_vel[:, :2]), dim=1)
        stable_rew = z_vel + ang_vel
        return stable_rew

    def leg_collision(self):
        # penalize legs ground touch
        leg_touch = (torch.norm(self.robot.contact_forces[:, self.robot.leg_indices, :], dim=-1) > 0.1)
        collision_rew = -1. * torch.sum(leg_touch.to(torch.float), dim=1)
        return collision_rew

    def smoothing_action(self):
        a0 = self.actions_recorder.get_last(0)
        a1 = self.actions_recorder.get_last(1)
        a2 = self.actions_recorder.get_last(2)
        first_order_smooth = torch.sum(torch.square(a1 - a0), dim=1)
        second_order_smooth = torch.sum(torch.square(a2 - 2 * a1 + a0), dim=1)
        smooth_rew = first_order_smooth + second_order_smooth
        return -0.005 * smooth_rew

    def torques_penalize(self):
        return -2e-5 * torch.sum(torch.square(self.robot.torques), dim=1)

    def sample_command(self, env_ids):
        self.command_buf[env_ids, 0] = torch_rand_float(self.cmd_lin_vel_x[0], self.cmd_lin_vel_x[1],
                                                        (len(env_ids), 1), device=self.device).squeeze(1)
        self.command_buf[env_ids, 1] = torch_rand_float(self.cmd_lin_vel_y[0], self.cmd_lin_vel_y[1],
                                                        (len(env_ids), 1), device=self.device).squeeze(1)
        self.command_buf[env_ids, 2] = torch_rand_float(self.cmd_ang_vel_yaw[0], self.cmd_ang_vel_yaw[1],
                                                        (len(env_ids), 1), device=self.device).squeeze(1)
        # self.command_buf[env_ids, 3] = torch_rand_float(self.cmd_base_height[0], self.cmd_base_height[1],
        #                                                 (len(env_ids), 1), device=self.device).squeeze(1)

    def update_terrain_curriculum(self, env_ids):
        if not self.isg_env.init_done:
            return
        distance = torch.norm(self.robot.base_pose[env_ids, :2] -
                              self.isg_env.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.isg_env.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.command_buf[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.isg_env.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.isg_env.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0))  # (the minimum level is zero)
        self.isg_env.update_terrain_level(env_ids, self.terrain_levels)


def get_args():
    import argparse
    parser = argparse.ArgumentParser("A1 conditional walking")
    parser.add_argument("--run-mode",
                        '-r',
                        type=str,
                        choices=['train', 'play', 'random'],
                        help="select either train play or random")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    run_args = get_args()

    run_policy(
        run_mode=run_args.run_mode,
        env_class=A1Conditional,
        env_cfg=A1EnvConfig(),
        policy_cfg=A1PPOConfig(),
        log_root=f"{LOG_ROOT}",
    )
