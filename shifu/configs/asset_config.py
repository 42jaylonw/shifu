import numpy as np

from .base_config import BaseConfig
from isaacgym import gymapi


class ActorConfig(BaseConfig):
    name = "DummyActor"
    root_dir = "./asset"
    urdf_filename = None
    default_pos = [0, 0, 0]
    default_quat = [0, 0, 0, 1]
    default_dof_pos = None
    domain_randomization = False
    dof_stiffness = None
    dof_damping = None

    def __init__(self):
        self._init_asset_options()
        super(ActorConfig, self).__init__()

    def _init_asset_options(self):
        asset_options = gymapi.AssetOptions()

        for attr in dir(self.asset_options):
            if '__' not in attr:
                setattr(asset_options, attr, eval(f'self.asset_options.{attr}'))

        # change the attribute type to isaac tasks asset_options
        self.asset_options = asset_options

    class asset_options:
        # essential
        fix_base_link = False
        default_dof_drive_mode = gymapi.DOF_MODE_NONE
        disable_gravity = False

        # modes
        collapse_fixed_joints = True
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        mesh_normal_mode = gymapi.FROM_ASSET
        use_physx_armature = True

        # numeric parameters
        thickness = 0.001
        # armature = 0.001
        # angular_damping = 0.
        # linear_damping = 0.
        # max_angular_velocity = 1000.
        # max_linear_velocity = 1000.


class BoxActorConfig(ActorConfig):
    box_dim = [0.05, 0.05, 0.05]
    mass = 0.1
    friction = 0.5
    color = [1., 1., 1.]

    class rigid_shape_props:
        friction = 1.0
        torsion_friction = 0.001
        restitution = 0.0


class ArmRobotActorConfig(ActorConfig):
    name = "DummyArmActor"
    root_dir = "./asset"
    urdf_filename = None
    default_pos = [0, 0, 0]
    default_quat = [0, 0, 0, 1]
    default_dof_pos = [0., 0., 0.5]
    end_effector_names = ['tip0']
    dof_stiffness = [400] * 5
    dof_damping = [80] * 5

    end_effector_velocity = 0.1
    min_ee_pos = [-0.25, -0.25, 0.11]
    max_ee_pos = [0.25, 0.25, 0.14]

    class asset_options(ActorConfig.asset_options):
        fix_base_link = True
        default_dof_drive_mode = gymapi.DOF_MODE_POS
        disable_gravity = True
        replace_cylinder_with_capsule = True


class LeggedRobotActorConfig(ActorConfig):
    name = "DummyArmActor"
    root_dir = "./asset"
    urdf_filename = None
    default_pos = [0, 0, 0]
    default_quat = [0, 0, 0, 1]
    default_dof_pos = [0.] * 12
    end_effector_names = ['foot0', 'foot1', 'foot2', 'foot3']
    dof_stiffness = [20] * 12
    dof_damping = [.5] * 12

    class asset_options(ActorConfig.asset_options):
        fix_base_link = False
        default_dof_drive_mode = gymapi.DOF_MODE_POS  # gymapi.DOF_MODE_POS
        disable_gravity = False
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
