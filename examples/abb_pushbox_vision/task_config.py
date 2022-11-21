from isaacgym import gymapi
from shifu.configs import (
    BaseEnvConfig,
    BoxActorConfig,
    ArmRobotActorConfig,
    CameraSensorConfig,
    PPOConfig
)

ASSET_ROOT = "./asset"


class TableConfig(BoxActorConfig):
    root_dir = ASSET_ROOT
    name = "table"
    default_pos = [0, 0, 0.05]
    default_quat = [0, 0, 0, 1]
    box_dim = [0.6, 0.6, 0.1]
    mass = 0.
    color = [0.8, 0.8, 0.8]

    class asset_options(BoxActorConfig.asset_options):
        fix_base_link = True


class PushBoxConfig(BoxActorConfig):
    root_dir = ASSET_ROOT
    name = "box"
    default_pos = [0, 0, 0.125]
    default_quat = [0, 0, 0, 1]
    box_dim = [0.05, 0.05, 0.05]
    mass = 0.1
    color = [.25, .65, .3]


class GoalBoxConfig(BoxActorConfig):
    root_dir = ASSET_ROOT
    name = "goal"
    default_pos = [0, 0, 0.1]
    default_quat = [0, 0, 0, 1]
    box_dim = [0.08, 0.08, 0.002]
    mass = 0.
    color = [0.8, 0., 0.]

    class asset_options(BoxActorConfig.asset_options):
        fix_base_link = True


class AbbRobotConfig(ArmRobotActorConfig):
    root_dir = ASSET_ROOT
    name = "AbbRobot-VacuumRod"
    urdf_filename = "urdf/abb_rod_description/urdf/abb_rod_isaac.urdf"
    end_effector_names = ['tip0']

    default_pos = [-0.48, 0, 0]
    default_quat = [0, 0, 0, 1]
    default_dof_pos = [0., 0.6437, 0.1748, 0., 0.7541, 0.]
    dof_stiffness = [800] * 6
    dof_damping = [40] * 6

    end_effector_velocity = 0.2  # [m/s]
    default_ee_quat = [0., 1., 0., 0]
    min_ee_pos = [-0.2, -0.2, 0.11]
    max_ee_pos = [0.2, 0.2, 0.14]  # 0.3


############################
#    Prior Stage Config    #
############################


class PriorStageEnvConfig(BaseEnvConfig):
    num_envs = 3000  #
    num_obs = 6  #
    num_privileged_obs = None
    num_actions = 3
    send_timeouts = True
    episode_length_s = 20.

    class sim(BaseEnvConfig.sim):
        dt = 0.02

    class control(BaseEnvConfig.control):
        decimation = int(0.1 / 0.02)

    class debug(BaseEnvConfig.debug):
        headless = True

    class normalization(BaseEnvConfig.normalization):
        clip_observations = 10.
        clip_actions = 1.


class PriorStagePPOConfig(PPOConfig):
    seed = 42
    runner_class_name = "AbbPushBoxTask"

    class policy(PPOConfig.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(PPOConfig.algorithm):
        schedule = 'adaptive'  # could be adaptive, fixed

    class runner(PPOConfig.runner):
        num_steps_per_env = 24
        max_iterations = 1500
        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'ppo_PushBox'
        run_name = 'AbbPushBox_PriorStage'
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt


#################################
#    Perceptual Stage Config    #
#################################


class PushBoxCameraConfig(CameraSensorConfig):
    name = 'rgbd_camera'
    local_lookat_positions = [[0.7, 0., 0.7], [0., 0., 0.1]]
    image_types = [
        gymapi.IMAGE_COLOR,
        gymapi.IMAGE_DEPTH,
        gymapi.IMAGE_SEGMENTATION
    ]
    image_normalization = True

    class camera_props(CameraSensorConfig.camera_props):
        """
        parameters from realsense d415
        """
        enable_tensors = True
        use_collision_geometry = False
        width = 128
        height = 128
        horizontal_fov = 42

        near_plane = 0.1  # [m]
        far_plane = 3  # [m]


###############################
#    Vision Stage Config      #
###############################


class VisionStageEnvConfig(PriorStageEnvConfig):
    num_envs = 1000
    num_obs = 6


class VisionStagePPOConfig(PPOConfig):
    seed = 42
    runner_class_name = "AbbPushBoxTask"

    class policy(PPOConfig.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(PPOConfig.algorithm):
        schedule = 'adaptive'  # could be adaptive, fixed

    class runner(PPOConfig.runner):
        num_steps_per_env = 24
        max_iterations = 2000
        # logging
        save_interval = 100
        experiment_name = 'ppo_PushBox'
        run_name = 'AbbPushBox_VisionStage'
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
