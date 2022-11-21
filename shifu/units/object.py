from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch

from shifu.units import Actor
from shifu.configs.asset_config import ActorConfig, BoxActorConfig


class Object(Actor):
    """
    only for static object
        todo: add more example objects
    """
    cfg: ActorConfig


class Box(Object):
    cfg: BoxActorConfig

    def __init__(
            self,
            cfg: BoxActorConfig,
    ):
        super(Box, self).__init__(cfg)

    def create_asset(self):
        self.asset = self.gym.create_box(self.sim,
                                         self.cfg.box_dim[0],
                                         self.cfg.box_dim[1],
                                         self.cfg.box_dim[2],
                                         self.asset_options)

    def load_to(self, env_id, env_handle, seg_id):
        super(Box, self).load_to(env_id, env_handle, seg_id)
        self.set_asset_rigid_properties(env_handle, mass=self.cfg.mass, friction=self.cfg.friction)
        self.gym.set_rigid_body_color(
            env_handle, self.actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*self.cfg.color))
