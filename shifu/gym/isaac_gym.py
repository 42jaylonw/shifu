import sys
from typing import Union, List

from isaacgym import gymapi, gymutil
from isaacgym import gymtorch

from shifu.configs import BaseEnvConfig, TerrainEnvConfig
from shifu.units import Unit, Object, Actor, Robot, Sensor, CameraSensor
from shifu.utils.torch_utils import free_tensor_attrs
from shifu.utils.terrain import Terrain, quat_apply_yaw

import numpy as np
import torch


class IsaacGymEnv:
    robot: Robot
    objects: List[Object]
    sensors: List[Sensor]

    def __init__(
            self,
            cfg: BaseEnvConfig,
    ):
        self.cfg = cfg
        self.dt = cfg.sim.dt * cfg.control.decimation
        self.decimation = cfg.control.decimation
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        self.spacing = cfg.spacing
        self.headless = cfg.debug.headless
        self.physics_engine = cfg.physics_engine
        self.sim_params = cfg.sim_params

        self._init_isaac_gym()

        self.viewer = None
        self.env_handles = []
        self._units: List[Unit] = []  # union of robot, objects, and sensors
        self._actors: List[Actor] = []  # union of robot and objects

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def step(self, action: torch.Tensor):
        self.render()
        self.robot.step(action)
        self.refresh_state()
        self.post_physics_step()

    def post_physics_step(self):
        pass

    def reset_idx(self, env_ids: Union[list, torch.Tensor], actors=None):
        if len(env_ids) == 0:
            return

        # default as reset all actors
        if actors is None:
            actors = self._actors

        root_state_indices = []
        for actor in actors:
            actor.reset_idx(env_ids)
            root_state_indices.append(actor.root_indices[env_ids])

        root_state_indices = torch.unique(torch.cat(root_state_indices)).to(dtype=torch.int32)

        # reset root tensor
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(root_state_indices),
                                                     len(root_state_indices))

    def create_envs(self, robot: Robot, objects: List[Object] = (), sensors: List[Sensor] = ()):
        self.init_done = False
        self.robot = robot
        self.objects = objects
        self.sensors = sensors

        # add units
        self._units.append(robot)
        self._units.extend(objects)
        self._units.extend(sensors)

        # add actors
        self._actors.append(robot)
        self._actors.extend(objects)

        for unit in self._units:
            unit.set_env(self)

        # create env handles
        env_lower = gymapi.Vec3(-self.spacing, -self.spacing, -self.spacing)
        env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
        for env_id in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            for seg_id, unit in enumerate(self._units, 1):
                unit.load_to(env_id, env_handle, seg_id)

            self.env_handles.append(env_handle)

        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self.init_done = True

    def _init_buffers(self):
        # initialize gym state tensors
        dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        contact_state = self.gym.acquire_net_contact_force_tensor(self.sim)

        # update state tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # dof
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        #  force sensors
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state)
        self.root_state = gymtorch.wrap_tensor(root_state)
        self.body_state = gymtorch.wrap_tensor(body_state)
        self.contact_state = gymtorch.wrap_tensor(contact_state)

        # initialize units buffers
        for unit in self._units:
            unit.init_buffers()

        if not self.headless:
            self._set_camera(self.cfg.debug.camera_pos, self.cfg.debug.camera_lookat)

    def refresh_state(self):
        self.gym.simulate(self.sim)  # updating states like actor_root_state
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # update state tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # dof
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        #  force sensors
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # Warning: if additional sensors contains in_step interactions, it must refresh here!
        # self.refresh_sensors()

    def refresh_sensors(self):
        for sensor in self.sensors:
            # for graphic sensors
            if isinstance(sensor, CameraSensor):
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                sensor.refresh()
                self.gym.end_access_image_tensors(self.sim)
            else:
                sensor.refresh()

    # def apply_randomization(self, env_ids):
    #
    #     for actor in self._actors:
    #         if actor.cfg.domain_randomization:
    #             for env_id in env_ids:
    #                 env_handle = self.env_handles[env_id]
    #                 actor.set_asset_rigid_properties(
    #                     env_handle,
    #                     actor.actor_handle,
    #                     mass=np.random.uniform(*actor.cfg.randomization.mass),
    #                     friction=np.random.uniform(*actor.cfg.randomization.friction))
    #     self.gym.simulate(self.sim)
    #     self.gym.refresh_actor_root_state_tensor(self.sim)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)

    def create_ground(self):
        """
        Create z-up plain
        Returns:
        """
        self.up_axis_idx = 2
        self._create_plane()
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

    def _create_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.
        self.gym.add_ground(self.sim, plane_params)

    def change_light(self, light_idx, direction, intensity=(0.5, 0.5, 0.5), ambient=(0.2, 0.2, 0.2)):
        self.gym.set_light_parameters(
            self.sim,
            light_idx,  # lightIndex 0-4 exceeds max lights: 0, 1, 2, 3, 4
            gymapi.Vec3(*intensity),  # intensity of the light focus in the range [0,1] per channel, in RGB.
            gymapi.Vec3(*ambient),  # ambient intensity of the ambient light in the range [0,1] per channel, in RGB.
            gymapi.Vec3(*direction)  # direction of the light focus
        )

    def _init_isaac_gym(self):
        self.gym = gymapi.acquire_gym()
        sim_device_type, sim_device_id = gymutil.parse_device_str(self.device)
        # graphics_device = -1 if self.headless else sim_device_id
        self.sim = self.gym.create_sim(sim_device_id,
                                       sim_device_id,
                                       self.physics_engine,
                                       self.sim_params)
        self.create_ground()

    def _set_camera(self, position, lookat):
        self.viewer = self.gym.create_viewer(
            self.sim, gymapi.CameraProperties())
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.cfg.debug.enable_viewer_sync = not self.cfg.debug.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.cfg.debug.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

            # render sensors
            for sensor in self.sensors:
                if isinstance(sensor, CameraSensor):
                    sensor.render()

            # viewer follow robot
            if self.cfg.debug.viewer_attach_robot_env_idx is not None:
                robot_pos = self.robot.base_pose[self.cfg.debug.viewer_attach_robot_env_idx, :3].cpu().detach().numpy()
                view_at = gymapi.Vec3(*robot_pos)
                view_from = gymapi.Vec3(*self.cfg.debug.camera_pos) + view_at
                self.gym.viewer_camera_look_at(self.viewer, None, view_from, view_at)

    def destroy(self):
        # destroy sensors
        for sensor in self.sensors:
            if isinstance(sensor, CameraSensor):
                for env_handle in self.env_handles:
                    self.gym.destroy_camera_sensor(self.sim, env_handle, sensor.camera_handle)
            free_tensor_attrs(sensor)

        # destroy envs
        for env_handle in self.env_handles:
            self.gym.destroy_env(env_handle)

        # destroy scenes
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

        # free torch memory
        free_tensor_attrs(self)


class TerrainGymEnv(IsaacGymEnv):
    cfg: TerrainEnvConfig

    def _init_buffers(self):
        super(TerrainGymEnv, self)._init_buffers()
        self.height_points = self._init_height_points()

    def create_envs(self, *args, **kwargs):
        # using custom origins
        self.spacing = 0
        super(TerrainGymEnv, self).create_envs(*args, **kwargs)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def post_physics_step(self):
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self.get_heights()

    def create_ground(self):
        assert isinstance(self.cfg, TerrainEnvConfig), f"cfg must be a TerrainEnvConfig"
        self.up_axis_idx = 2
        if self.cfg.terrain.mesh_type == 'heightfield':
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
            self._create_heightfield()
        elif self.cfg.terrain.mesh_type == 'trimesh':
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
            self._create_trimesh()
        else:
            raise NotImplementedError("cfg.terrain.mesh_type must be one of heightfield or trimesh")

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # put robots at the origins defined by the terrain
        max_init_level = self.cfg.terrain.max_init_terrain_level
        if not self.cfg.terrain.curriculum:
            max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                       (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
            torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def update_terrain_level(self, env_ids, levels):
        if not self.init_done:
            return
        self.terrain_levels = levels
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.robot.base_pose[env_ids, 3:7].repeat(1, self.num_height_points),
                self.height_points[env_ids]) + (self.robot.base_pose[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.robot.base_pose[:, 3:7].repeat(1, self.num_height_points),
                self.height_points) + (self.robot.base_pose[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
