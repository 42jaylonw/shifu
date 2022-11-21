from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch

from shifu.units import Actor
from shifu.configs.asset_config import ActorConfig, ArmRobotActorConfig, LeggedRobotActorConfig


class Robot(Actor):
    cfg: ActorConfig
    default_dof_pos: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    dof_targets: torch.Tensor

    def __init__(
            self,
            cfg: ActorConfig,
    ):
        super(Robot, self).__init__(cfg=cfg)
        self.cfg = cfg

    def reset_idx(self, env_ids):
        self._reset_dof_state(env_ids)
        self._reset_root_state(env_ids)

    def step(self, actions):
        self._internal_motor_step(actions)

    def _init_props(self):
        super(Robot, self)._init_props()

        self.dof_props['driveMode'][:] = self.asset_options.default_dof_drive_mode
        self.dof_props['stiffness'] = self.cfg.dof_stiffness
        self.dof_props['damping'] = self.cfg.dof_damping

        self.dof_lower_limits = to_torch(self.dof_props['lower'], device=self.device)
        self.dof_upper_limits = to_torch(self.dof_props['upper'], device=self.device)
        self.dof_vel_limits = to_torch(self.dof_props['velocity'], device=self.device)
        self.torque_limits = to_torch(self.dof_props['effort'], device=self.device)

    def load_to(self, env_id, env_handle, seg_id):
        super(Robot, self).load_to(env_id, env_handle, seg_id)
        self.gym.set_actor_dof_properties(env_handle, self.actor_handle, self.dof_props)

    def init_buffers(self):
        super(Robot, self).init_buffers()
        self.default_dof_pos = to_torch(self.cfg.default_dof_pos, device=self.device)
        self.dof_pos = self.env.dof_state.view(self.env.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.env.dof_state.view(self.env.num_envs, self.num_dof, 2)[..., 1]
        self.dof_targets = torch.zeros((self.env.num_envs, self.num_dof), dtype=torch.float, device=self.device)

    def _internal_motor_step(self, action):
        # GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        if self.asset_options.default_dof_drive_mode == gymapi.DOF_MODE_EFFORT:
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(action))
        elif self.asset_options.default_dof_drive_mode == gymapi.DOF_MODE_POS:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action))
        elif self.asset_options.default_dof_drive_mode == gymapi.DOF_MODE_VEL:
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(action))
        else:
            raise NotImplementedError

    def apply_dof_targets(self, dof_targets):
        for i in range(int(self.env.decimation)):
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_targets))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

    def _reset_dof_state(self, env_ids):
        self.dof_targets[env_ids] = self.default_dof_pos.clone()
        self.dof_pos[env_ids] = self.default_dof_pos.clone()
        self.dof_vel[env_ids] = 0.
        body_indices = self.root_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_targets),
                                                        gymtorch.unwrap_tensor(body_indices),
                                                        len(body_indices))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.env.dof_state),
                                              gymtorch.unwrap_tensor(body_indices),
                                              len(env_ids))

    @property
    def base_pose(self):
        return self.env.root_state[self.root_indices, :7]

    def get_root_state(self):
        return self.env.root_state[self.root_indices]

    def set_root_state(self, root_state):
        """
        apply robot's root state to the simulation
        """
        self.env.root_state[self.root_indices] = root_state
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.env.root_state))


class ArmRobot(Robot):
    cfg: ArmRobotActorConfig
    num_ee: int

    def __init__(
            self,
            cfg: ArmRobotActorConfig,
    ):
        super(ArmRobot, self).__init__(cfg)
        self.cfg = cfg
        self.end_effector_names = cfg.end_effector_names
        self.end_effector_velocity = cfg.end_effector_velocity

    def init_buffers(self):
        super(ArmRobot, self).init_buffers()
        # only support one end effector for now
        ee_indices = [self.rigid_body_dict[n] for n in self.cfg.end_effector_names]
        self.ee_indices = to_torch(ee_indices, dtype=torch.long, device=self.device)
        self.num_ee = len(ee_indices)
        self.contact_forces = self.env.contact_state.view(self.env.num_envs, -1, 3)
        self.ee_pose_targets = torch.zeros((self.env.num_envs, 7), dtype=torch.float, device=self.device)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.name)
        self.gym.refresh_jacobian_tensors(self.sim)
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self.j_ee = jacobian[:, ee_indices[0] - 1]  # only support one ee for now

    def load_to(self, env_id, env_handle, seg_id):
        super(ArmRobot, self).load_to(env_id, env_handle, seg_id)
        self.set_segmentation_id(env_handle, seg_id)

    def apply_target_end_positions(self, tar_pose):
        self.dof_targets[:] = self.inverse_kinematics(tar_pose)
        self.apply_dof_targets(self.dof_targets)

    @property
    def body_state(self):
        return self.env.body_state.view(
            self.env.num_envs, -1, 13
        )[:, :self.num_bodies].view(self.env.num_envs, self.num_bodies, -1)

    @property
    def ee_pose(self):
        return self.body_state[:, self.ee_indices, :7]

    @property
    def ee_vel(self):
        return self.body_state[:, self.ee_indices, 7:]

    @property
    def ee_forces(self):
        return self.contact_forces[:, self.ee_indices]

    @staticmethod
    def orientation_error(desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def inverse_kinematics(self, goal_pose, damping=0.05):
        # todo: find a smarter way to set damping
        ee_pos = self.ee_pose[:, 0, :3]
        ee_quat = self.ee_pose[:, 0, 3:7]

        # compute position and orientation error
        goal_pos = goal_pose[:, :3]
        goal_quat = goal_pose[:, 3:7]

        pos_err = goal_pos - ee_pos
        orn_err = self.orientation_error(goal_quat, ee_quat)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # solve damped least squares
        j_eef_T = torch.transpose(self.j_ee, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_ee @ j_eef_T + lmbda) @ dpose).view(
            self.env.num_envs, self.num_dof)

        tar_dof_pos = self.dof_pos + u
        return tar_dof_pos


class LeggedRobot(ArmRobot):
    cfg: LeggedRobotActorConfig
    num_ee: int

    def __init__(self, cfg: LeggedRobotActorConfig):
        super(ArmRobot, self).__init__(cfg)
        self.cfg = cfg
        self.end_effector_names = cfg.end_effector_names
        self.ee_indices = []

    def init_buffers(self):
        super(ArmRobot, self).init_buffers()
        # end effectors
        ee_indices = [self.rigid_body_dict[n] for n in self.cfg.end_effector_names]
        self.ee_indices = to_torch(ee_indices, dtype=torch.long, device=self.device)
        self.num_ee = len(ee_indices)
        self.contact_forces = self.env.contact_state.view(self.env.num_envs, -1, 3)

        # jacobian entries corresponding to robot
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.name)
        self.gym.refresh_jacobian_tensors(self.sim)
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self.j_ee = jacobian[:, ee_indices]

        # rigid body
        self.gravity_vec = to_torch(get_axis_params(-1., self.env.up_axis_idx), device=self.device).repeat(
            (self.env.num_envs, 1))
        self.base_lin_vel = quat_rotate_inverse(self.base_pose[:, 3:7], self.env.root_state[self.root_indices, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_pose[:, 3:7],
                                                self.env.root_state[self.root_indices, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_pose[:, 3:7], self.gravity_vec)

    def step(self, actions):
        self.dof_targets[:] = self.dof_pos[:, :self.num_dof] + actions
        self.apply_dof_targets(self.dof_targets)
        self.post_step()

    def post_step(self):
        self.gravity_vec[:] = to_torch(get_axis_params(-1., self.env.up_axis_idx), device=self.device).repeat(
            (self.env.num_envs, 1))
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_pose[:, 3:7],
                                                   self.env.root_state[self.root_indices, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_pose[:, 3:7],
                                                   self.env.root_state[self.root_indices, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_pose[:, 3:7], self.gravity_vec)

    def apply_force_on_base(self, force_tensor, pos_tensor=None):
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(force_tensor),
            gymtorch.unwrap_tensor(pos_tensor) if pos_tensor is not None else None
        )
