import torch
from shifu.runner import datetime_logdir
from shifu.runner.utils import class_to_dict, set_seed, get_load_path
from rsl_rl.runners import OnPolicyRunner


class _OnPolicyRunner(OnPolicyRunner):
    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)    # load in different devices
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']


def run_policy(run_mode, env_class, env_cfg, policy_cfg, log_root="./logs",
               play_num_envs=50, play_iterations=3000):
    if run_mode == 'train':
        env = env_class(env_cfg)
        ppo_runner = build_policy_runner(env, policy_cfg, log_root)
        ppo_runner.learn(num_learning_iterations=policy_cfg.runner.max_iterations, init_at_random_ep_len=True)
    elif run_mode == 'play':
        env_cfg.num_envs = play_num_envs
        env_cfg.debug.headless = False
        env = env_class(env_cfg)
        policy = load_policy(env, policy_cfg, log_root)
        env.reset()
        obs = env.get_observations()
        for i in range(play_iterations):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
    elif run_mode == 'random':
        env_cfg.num_envs = play_num_envs
        env_cfg.debug.headless = False
        env = env_class(env_cfg)

        env.reset()
        for i in range(play_iterations):
            actions = 2 * torch.rand(env.num_envs, env.num_actions, device=env.device) - 1
            obs, _, rews, dones, infos = env.step(actions.detach())
    else:
        raise NotImplementedError


def load_policy(env, policy_cfg, log_root, device='cuda:0'):
    ppo_runner = build_policy_runner(env, policy_cfg, log_root, resume=True, device=device)
    policy = ppo_runner.get_inference_policy()
    return policy


def build_policy_runner(
        env,
        train_cfg,
        log_root="./logs",
        device="cuda:0",
        resume=False,
):
    log_dir = datetime_logdir(log_root, train_cfg.runner.run_name)
    train_cfg_dict = class_to_dict(train_cfg)

    set_seed(train_cfg.seed)
    runner = _OnPolicyRunner(env, train_cfg_dict, log_dir, device=device)

    if resume:
        # load previously trained model
        resume_path = get_load_path(log_root,
                                    load_run=train_cfg.runner.load_run,
                                    checkpoint=train_cfg.runner.checkpoint)
        print(f"Loading model from: {resume_path}")
        runner.load(resume_path)

    return runner
