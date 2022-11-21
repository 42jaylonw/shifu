from isaacgym import gymapi
from shifu.configs import (
    TerrainEnvConfig,
    LeggedRobotActorConfig,
    PPOConfig
)

ASSET_ROOT = "./asset"


class A1ActorConfig(LeggedRobotActorConfig):
    name = "a1_robot"
    root_dir = ASSET_ROOT
    urdf_filename = "urdf/a1/urdf/a1.urdf"
    default_pos = [0, 0, 0.42]  # 0.42
    default_quat = [0, 0, 0, 1.]
    default_dof_pos = [0.1, 0.8, -1.5,
                       0.1, 0.8, -1.5,
                       -0.1, 0.8, -1.5,
                       -0.1, 0.8, -1.5]
    end_effector_names = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
    dof_stiffness = [20] * 12
    dof_damping = [.5] * 12

    class asset_options(LeggedRobotActorConfig.asset_options):
        default_dof_drive_mode = gymapi.DOF_MODE_EFFORT  # gymapi.DOF_MODE_POS


class A1EnvConfig(TerrainEnvConfig):
    num_envs = 4000
    num_obs = 259
    num_privileged_obs = None
    num_actions = 12
    num_actions_history = 3
    send_timeouts = True
    episode_length_s = 10.

    class sim(TerrainEnvConfig.sim):
        dt = 0.005

    class control(TerrainEnvConfig.control):
        decimation = 4

    class debug(TerrainEnvConfig.debug):
        headless = True
        camera_pos = [1., -1., 1.]

    class normalization(TerrainEnvConfig.normalization):
        clip_observations = 100.
        clip_actions = 1.

    class terrian(TerrainEnvConfig.terrain):
        mesh_type = 'trimesh'
        num_rows = 3  # number of terrain rows (levels) 0-10
        num_cols = 20  # number of terrain cols (types)  0-20
        max_init_terrain_level = 3  # starting curriculum state


class A1PPOConfig(PPOConfig):
    seed = 42
    runner_class_name = "A1Conditional"

    class runner(PPOConfig.runner):
        num_steps_per_env = 24
        max_iterations = 3000
        # logging
        save_interval = 100
        experiment_name = 'commands_and_terrain'
        run_name = 'ppo_A1Conditional'
