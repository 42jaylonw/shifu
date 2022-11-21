from .base_config import BaseConfig
from isaacgym import gymapi


class BaseEnvConfig(BaseConfig):
    num_envs = 5
    num_obs = 10
    num_privileged_obs = None  # critic obs for asymmetric training
    num_actions = 3
    num_actions_history = None
    send_timeouts = True  # send time out information to the algorithm
    episode_length_s = 20  # 20 # episode length in seconds

    spacing = 1.
    device = 'cuda:0'
    physics_engine = gymapi.SIM_PHYSX

    def __init__(self):
        self._init_sim_params()
        super(BaseEnvConfig, self).__init__()

    def _init_sim_params(self):
        # get default set of parameters
        sim_params = gymapi.SimParams()

        for attr in dir(self.sim):
            if '__' not in attr:
                if 'physx' == attr:
                    for physx_attr in dir(self.sim.physx):
                        if '__' not in physx_attr:
                            setattr(sim_params.physx, physx_attr, eval(f'self.sim.physx.{physx_attr}'))
                else:
                    setattr(sim_params, attr, eval(f'self.sim.{attr}'))

        # change the attribute type to isaac tasks asset_options
        self.sim_params = sim_params

    class sim:
        # set common parameters
        dt = 0.005
        substeps = 1
        up_axis = gymapi.UP_AXIS_Z
        gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        use_gpu_pipeline = True

        class physx:
            # set PhysX-specific parameters
            num_threads = 10
            use_gpu = True
            solver_type = 1
            num_position_iterations = 8
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class debug:
        headless = False
        camera_pos = [1., -1., 1.]
        camera_lookat = [0, 0, 0]
        enable_viewer_sync = True
        viewer_attach_robot_env_idx = None

    class normalization:
        clip_observations = 100.
        clip_actions = 1.

    class control:
        decimation = 4


class TerrainEnvConfig(BaseEnvConfig):
    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels) 0-10
        num_cols = 20  # number of terrain cols (types)  0-20
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

        curriculum = True
        max_init_terrain_level = 5  # starting curriculum state
