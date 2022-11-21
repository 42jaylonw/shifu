import typing

import numpy as np
import torch

from rsl_rl.env import VecEnv
from shifu.gym.isaac_gym import IsaacGymEnv, TerrainGymEnv
from shifu.configs import BaseEnvConfig, TerrainEnvConfig
from shifu.utils.torch_utils import free_tensor_attrs
from shifu.utils.train import HistoryRecorder


# TODO: [
#  make a _set_env(): for user load their robot, objects, and sensors
#  ]


class ShifuVecEnv(VecEnv):
    def __init__(
            self,
            cfg: BaseEnvConfig,
    ):
        self.cfg = cfg

        if isinstance(cfg, TerrainEnvConfig):
            self.isg_env = TerrainGymEnv(cfg)
        else:
            self.isg_env = IsaacGymEnv(cfg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.num_envs = self.isg_env.num_envs
        self.device = self.isg_env.device
        self.num_obs = cfg.num_obs
        self.num_privileged_obs = cfg.num_privileged_obs
        self.num_actions = cfg.num_actions
        self.clip_obs = cfg.normalization.clip_observations
        self.clip_actions = cfg.normalization.clip_actions
        self.max_episode_length_s = cfg.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.isg_env.dt)

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float,
                                   requires_grad=False)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}

        if self.cfg.num_actions_history:
            self.actions_recorder = HistoryRecorder(self.actions.shape, self.cfg.num_actions_history,
                                                    device=self.device)

        self.privileged_obs_buf = None if self.num_privileged_obs is None \
            else torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)

        self.common_step_counter = 0

        self.reward_functions = self.build_reward_functions()
        self._prepare_reward_functions()

    def create_pybullet_twin(self):
        from shifu.gym.pybullet_twin import PybulletTwinEnv
        self.pyb_env = PybulletTwinEnv(dt=self.isg_env.dt,
                                       robot_cfg=self.isg_env.robot.cfg,
                                       obj_cfg_list=[obj.cfg for obj in self.isg_env.objects],
                                       sensor_cfg_list=[snr.cfg for snr in self.isg_env.sensors],
                                       decimation=self.isg_env.decimation)

    def destroy(self):
        # free all tensors
        self.isg_env.destroy()
        free_tensor_attrs(self)

    def build_reward_functions(self) -> typing.List:
        # add your reward functions
        raise NotImplementedError

    def compute_observations(self):
        raise NotImplementedError

    def step(self, actions: torch.Tensor):
        assert self.isg_env.robot, "add robot before step"
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        self.isg_env.step(self.actions)
        self.post_step()
        self.obs_buf = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_step(self):
        # update observation, reward, done, and extras buffers
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.compute_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()
        self.isg_env.refresh_sensors()
        if self.cfg.num_actions_history:
            self.actions_recorder.add(self.actions)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, pri_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, pri_obs

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self.isg_env.reset_idx(env_ids)

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        if self.cfg.num_actions_history:
            self.actions_recorder.reset_idx(env_ids)

        # fill extras
        self.extras["episode"] = {}

        self.log_info(env_ids)

        if self.cfg.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def episode_log(self, env_ids) -> typing.Dict:
        """
        Args:
            env_ids: ids of env ready to reset

        Examples:

            log mean success rate of the episode

            def episode_log(self, env_ids):
                return {'success_rate': torch.mean(self.success_buf.to(torch.float)[env_ids])}

        Returns:
            A custom logging dictionary
        """
        pass

    def log_info(self, env_ids):
        # log all rewards
        for key in self.episode_rewards.keys():
            self.extras["episode"][key] = torch.mean(self.episode_rewards[key][env_ids]) / self.max_episode_length_s
            self.episode_rewards[key][env_ids] = 0.

        ep_info = self.episode_log(env_ids)
        if ep_info:
            for key, value in ep_info.items():
                self.extras["episode"][key] = value

    def _prepare_reward_functions(self):
        assert len(self.reward_functions) > 0
        self.episode_rewards = {}
        for rew_func in self.reward_functions:
            self.episode_rewards[rew_func.__name__] = torch.zeros(self.isg_env.num_envs,
                                                                  device=self.isg_env.device,
                                                                  dtype=torch.float)

    def compute_termination(self):
        """
        Examples:
            def compute_termination(self):
                self.time_out_buf = self.episode_length_buf > self.max_episode_length
                self.success_buf = self.is_success().to(torch.bool)
                self.reset_buf = self.time_out_buf | self.success_buf
        Returns:
            None
        """
        raise NotImplementedError

    def compute_reward(self):
        self.rew_buf[:] = 0.
        for rew_func in self.reward_functions:
            rew = rew_func()
            self.episode_rewards[rew_func.__name__] += rew
            self.rew_buf[:] += rew

        # self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.) # non-negative

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf
