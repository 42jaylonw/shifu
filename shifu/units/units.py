from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from shifu.configs import BaseConfig, ActorConfig, BaseSensorConfig

import torch


class Unit(object):
    cfg: BaseConfig
    gym: gymapi.Gym
    sim: gymapi.Sim
    device: [str, torch.device]

    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        self.name = cfg.name

    def set_env(self, env):
        self.env = env
        self.gym = env.gym
        self.sim = env.sim
        self.device = env.device
        self._init_props()

    def _init_props(self):
        raise NotImplementedError

    def reset_idx(self, env_ids):
        raise NotImplementedError

    def load_to(self, env_id, env_handle, seg_id):
        raise NotImplementedError

    def init_buffers(self):
        raise NotImplementedError


class Actor(Unit):
    cfg: ActorConfig
    asset: gymapi.Asset
    actor_handle: int
    segmentation_id: int
    rigid_body_dict: dict
    default_base_pose: torch.Tensor

    def __init__(self, cfg: ActorConfig):
        super(Actor, self).__init__(cfg)
        self.asset_options = cfg.asset_options
        self.root_indices = []
        self.rigid_body_dict = {}

    def reset_idx(self, env_ids):
        self._reset_root_state(env_ids)

    def load_to(self, env_id, env_handle, seg_id):
        origin = self.env.env_origins[env_id].clone()
        self._init_root_pose.p += gymapi.Vec3(*origin)

        # apply randomization
        try:
            rand_rigid_shape_props = self.random_rigid_shape_props(env_id, self.default_rigid_shape_props)
            self.gym.set_asset_rigid_shape_properties(self.asset, rand_rigid_shape_props)
        except NotImplementedError:
            pass

        self.actor_handle = self.gym.create_actor(env_handle, self.asset, self._init_root_pose, self.name, env_id, 0)
        body_index = self.gym.get_actor_index(env_handle, self.actor_handle, gymapi.DOMAIN_SIM)
        self.root_indices.append(body_index)
        self.set_segmentation_id(env_handle, seg_id)

    def create_asset(self):
        self.asset = self.gym.load_asset(self.sim,
                                         self.cfg.root_dir,
                                         self.cfg.urdf_filename,
                                         self.asset_options)

    def _init_props(self):
        self._init_root_pose = gymapi.Transform()
        self._init_root_pose.p = gymapi.Vec3(*self.cfg.default_pos)
        self._init_root_pose.r = gymapi.Quat(*self.cfg.default_quat)

        self.create_asset()
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self.default_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.asset)

        self.num_dof = self.gym.get_asset_dof_count(self.asset)
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)

    def set_segmentation_id(self, env_handle, seg_id):
        self.segmentation_id = seg_id
        self.rigid_body_dict = self.gym.get_actor_rigid_body_dict(env_handle, self.actor_handle)
        for rigid_name, rigid_id in self.rigid_body_dict.items():
            self.gym.set_rigid_body_segmentation_id(env_handle, self.actor_handle, rigid_id, seg_id)

    def set_asset_rigid_properties(self, env_handle, mass=None, friction=None):
        if friction is not None:
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, self.actor_handle)
            for shape_prop in shape_props:
                shape_prop.friction = friction
            self.gym.set_actor_rigid_shape_properties(env_handle, self.actor_handle, shape_props)

        if mass is not None:
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, self.actor_handle)
            for body_prop in body_props:
                body_prop.mass = mass
            # Note: Changing the center-of-mass when using the GPU pipeline
            # is currently not supported (but mass and inertia can be changed).
            self.gym.set_actor_rigid_body_properties(env_handle, self.actor_handle, body_props, recomputeInertia=True)

    def random_rigid_shape_props(self, env_ids, rigid_shape_props):
        """
        apply custom random body properties, like friction, restitution, etc.
        find more details about isaacgym.gymapi.RigidShapeProperties on isaacgym documents.
        Args:
            env_ids: environment indices
            rigid_shape_props: rigid shape property from IsaacGym

        Returns:
            rigid_shape_props
        """
        raise NotImplementedError

    def init_buffers(self):
        self.root_indices = to_torch(self.root_indices, dtype=torch.long, device=self.device)
        self.rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset)
        self.default_base_pose = to_torch(self.cfg.default_pos + self.cfg.default_quat, device=self.device)

    def _reset_root_state(self, env_ids):
        robot_indices = self.root_indices[env_ids]
        self.env.root_state[robot_indices, :3] = self.default_base_pose[:3] + self.env.env_origins[env_ids]
        self.env.root_state[robot_indices, 3:7] = self.default_base_pose[3:7]
        self.env.root_state[robot_indices, 7:] = 0.  # [7:10]: lin vel, [10:13]: ang vel

    @property
    def base_pose(self):
        return self.env.root_state[self.root_indices, :7]


class Sensor(Unit):
    cfg: BaseSensorConfig

    def __init__(self, cfg: BaseSensorConfig):
        super(Sensor, self).__init__(cfg)

    def _init_props(self):
        raise NotImplementedError

    def reset_idx(self, env_ids):
        raise NotImplementedError

    def load_to(self, env_id, env_handle, seg_id):
        raise NotImplementedError

    def refresh(self):
        raise NotImplementedError

    def init_buffers(self):
        raise NotImplementedError
