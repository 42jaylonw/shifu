from examples.a1_conditional.a1_conditional import A1Conditional, A1EnvConfig, A1PPOConfig, LOG_ROOT
from shifu.runner import load_policy

from shifu.utils.play import ControlPanel
import torch


def play_with_command_panel(num_env_play=50, play_duration=120, terrain_level=2):
    env_cfg = A1EnvConfig()
    policy_cfg = A1PPOConfig()
    env_cfg.num_envs = num_env_play
    env_cfg.debug.headless = False
    env_cfg.debug.viewer_attach_robot_env_idx = 5
    env = A1Conditional(env_cfg)
    policy = load_policy(env, policy_cfg, LOG_ROOT)
    env.reset()
    obs = env.get_observations()

    # reset the terrain levels
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    env.terrain_levels = torch.ones_like(all_env_ids) * terrain_level
    env.isg_env.update_terrain_level(all_env_ids, env.terrain_levels)

    # build a controller
    controller = ControlPanel(num_parameters=4)
    controller.add_parameter('lin_vel_x', -1., 1.)
    controller.add_parameter('lin_vel_y', -1., 1.)
    controller.add_parameter('ang_vel_yaw', -1., 1.)

    for i in range(int(play_duration/env.isg_env.dt)):
        obs = obs.detach()
        # apply command
        cmd_lin_vel_x = controller.get_value('lin_vel_x')
        cmd_lin_vel_y = controller.get_value('lin_vel_y')
        cmd_ang_vel_yaw = controller.get_value('ang_vel_yaw')
        cmd_tensor = torch.Tensor([cmd_lin_vel_x, cmd_lin_vel_y, cmd_ang_vel_yaw]).to(env.device)
        obs[:, :3] = cmd_tensor

        actions = policy(obs)
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    play_with_command_panel()
