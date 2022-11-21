import enum

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from shifu.units import Sensor
from shifu.configs import CameraSensorConfig
from shifu.utils.image import seg2rgb, normalize_color

# todo: [
#  depth,
#  point cloud,
#  semantic segmentation
#  ]

IMAGE_TYPE_COLOR = gymapi.IMAGE_COLOR
IMAGE_TYPE_DEPTH = gymapi.IMAGE_DEPTH
IMAGE_TYPE_SEGMENTATION = gymapi.IMAGE_SEGMENTATION
IMAGE_TYPE_OPTICAL_FLOW = gymapi.IMAGE_OPTICAL_FLOW


# todo: add normalization to all other image types


def normalize_optical_flow(optical_flow, width, height):
    optical_flow_in_pixels = torch.zeros_like(optical_flow)
    # Horizontal (u)
    optical_flow_in_pixels[0, 0] = width * (optical_flow[0, 0] / 2 ** 15)
    # Vertical (v)
    optical_flow_in_pixels[0, 1] = height * (optical_flow[0, 1] / 2 ** 15)


class CameraPose(enum.Enum):
    LocalLookat = 0
    Transform = 1
    AttachLocalTransform = 2


# ${IsaacGymHome}/docs/programming/graphics.html?highlight=sensors#camera-sensors
class CameraSensor(Sensor):
    cfg: CameraSensorConfig
    camera_handle: int
    color_buf: torch.Tensor
    depth_buf: torch.Tensor
    segmentation_buf: torch.Tensor
    optical_flow_buf: torch.Tensor
    proj_matrix: np.matrix
    view_matrix: np.matrix

    def __init__(
            self,
            cfg: CameraSensorConfig,
    ):
        # todo: add param to cfg defines Depth and Semantic camera sensor
        super(CameraSensor, self).__init__(cfg)
        self.width = cfg.camera_props.width
        self.height = cfg.camera_props.height
        self.near_plane = cfg.camera_props.near_plane
        self.far_plane = cfg.camera_props.far_plane

    def init_buffers(self):
        for img_type in self.cfg.image_types:
            if img_type == IMAGE_TYPE_COLOR:
                # 4 x 8 bit unsigned int - RGBA color.
                # 3 x float - normalized RGB color [0 - 1]
                self.color_buf = torch.zeros(
                    self.env.num_envs, self.height, self.width,
                    3 if self.cfg.image_normalization else 4,
                    dtype=torch.float if self.cfg.image_normalization else torch.uint8,
                    device=self.device)
            elif img_type == IMAGE_TYPE_DEPTH:
                # float - negative distance from camera to pixel in view direction in world coordinate units (meters).
                self.depth_buf = torch.zeros(
                    self.env.num_envs, self.height, self.width,
                    dtype=torch.float, device=self.device)
            elif img_type == IMAGE_TYPE_SEGMENTATION:
                # 32bit unsigned int - ground truth semantic segmentation of each pixel.
                self.segmentation_buf = torch.zeros(
                    self.env.num_envs, self.height, self.width,
                    dtype=torch.int32,
                    device=self.device)
            elif img_type == IMAGE_TYPE_OPTICAL_FLOW:
                # 2x 16bit signed int - screen space motion vector per pixel, normalized.
                # ??? 1x 16bit now ???
                self.optical_flow_buf = torch.zeros(
                    self.env.num_envs, self.height, self.width,
                    dtype=torch.int16,
                    device=self.device)
            else:
                raise NotImplementedError

    def _init_props(self):
        if self.cfg.local_lookat_positions is not None:
            assert self.cfg.transform is None and self.cfg.attach_local_transform is None
            self.local_lookat_position = (gymapi.Vec3(*self.cfg.local_lookat_positions[0]),
                                          gymapi.Vec3(*self.cfg.local_lookat_positions[1]))
            self._pose_type = CameraPose.LocalLookat
        elif self.cfg.transform is not None:
            assert self.cfg.local_lookat_positions is None and self.cfg.attach_local_transform is None
            self.transform = gymapi.Transform()
            self.transform.p = gymapi.Vec3(*self.cfg.transform[0])
            self.transform.r = gymapi.Quat(*self.cfg.transform[1])
            self._pose_type = CameraPose.Transform
        elif self.cfg.attach_local_transform is not None:
            # todo: add attach method
            raise NotImplementedError('Currently not support')
        else:
            raise NotImplementedError(
                'choose one of method from local_lookat_positions and transform')

    def reset_idx(self, env_ids):
        pass

    def load_to(self, env_id, env_handle, seg_id):
        camera_handle = self.gym.create_camera_sensor(env_handle, self.cfg.camera_props)

        if self._pose_type == CameraPose.LocalLookat:
            self.gym.set_camera_location(camera_handle,
                                         env_handle,
                                         self.local_lookat_position[0],
                                         self.local_lookat_position[1])
        elif self._pose_type == CameraPose.Transform:
            self.gym.set_camera_transform(camera_handle, env_handle, self.transform)
        else:
            raise NotImplementedError

        if env_id == 0:
            self.camera_handle = camera_handle
            self.proj_matrix = np.matrix(
                self.gym.get_camera_proj_matrix(self.sim, env_handle, self.camera_handle))
            self.view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, env_handle, self.camera_handle))

    def set_camera_transform(self, position, rotation):
        for env_id, env_handle in enumerate(self.env.env_handles):
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(*position)
            transform.r = gymapi.Quat(*rotation)
            self.gym.set_camera_transform(self.camera_handle, env_handle, self.transform)
            if env_id == 0:
                self.transform = transform
                self.proj_matrix = np.matrix(
                    self.gym.get_camera_proj_matrix(self.sim, env_handle, self.camera_handle))
                self.view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, env_handle, self.camera_handle))

    def set_camera_location(self, local_pos, lookat_pos):
        # currently only support change all camera parm at once
        for env_id, env_handle in enumerate(self.env.env_handles):
            self.gym.set_camera_location(self.camera_handle, env_handle, gymapi.Vec3(*local_pos),
                                         gymapi.Vec3(*lookat_pos))
            if env_id == 0:
                self.local_lookat_position = (local_pos, lookat_pos)
                self.proj_matrix = np.matrix(
                    self.gym.get_camera_proj_matrix(self.sim, env_handle, self.camera_handle))
                self.view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, env_handle, self.camera_handle))

    def refresh(self):
        self.refresh_image_tensors()

    def refresh_image_tensors(self):
        for i in range(self.env.num_envs):
            env_handle = self.env.env_handles[i]

            if IMAGE_TYPE_COLOR in self.cfg.image_types:
                color_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, self.camera_handle, IMAGE_TYPE_COLOR))
                self.color_buf[i, ...] = normalize_color(color_tensor) if self.cfg.image_normalization else color_tensor

            if IMAGE_TYPE_DEPTH in self.cfg.image_types:
                # Isaac gives negative depth map !
                depth_tensor = - gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, self.camera_handle, IMAGE_TYPE_DEPTH))
                self.depth_buf[i, ...] = depth_tensor

            if IMAGE_TYPE_SEGMENTATION in self.cfg.image_types:
                self.segmentation_buf[i, ...] = gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, self.camera_handle,
                                                         IMAGE_TYPE_SEGMENTATION))

            if IMAGE_TYPE_OPTICAL_FLOW in self.cfg.image_types:
                self.optical_flow_buf[i, ...] = gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, self.camera_handle,
                                                         IMAGE_TYPE_OPTICAL_FLOW))

    def render(self, render_idx=0, dsize=None):
        if dsize is None:
            dsize = (256 * 4, int(256 * self.height / self.width))
        render_img = np.zeros((self.height, 4 * self.width, 3))
        for img_type in self.cfg.image_types:
            if img_type == IMAGE_TYPE_COLOR:
                color_img = self.color_buf[render_idx] if self.cfg.image_normalization \
                    else normalize_color(self.color_buf[render_idx])
                color_img = cv2.cvtColor(color_img.detach().cpu().numpy(), cv2.COLOR_RGBA2BGR)
                render_img[:, :self.width] = color_img

            elif img_type == IMAGE_TYPE_DEPTH:
                depth_img = self.depth_buf[render_idx].detach().cpu().numpy()
                depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
                depth_img = plt.get_cmap('viridis')(depth_img)[..., :3].astype(np.float32)
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_RGBA2BGR)
                render_img[:, self.width:2 * self.width] = depth_img

            elif img_type == IMAGE_TYPE_SEGMENTATION:
                seg = self.segmentation_buf[render_idx].to(torch.float).detach().cpu().numpy()
                render_img[:, 2 * self.width:3 * self.width] = cv2.cvtColor(seg2rgb(seg), cv2.COLOR_RGB2BGR)
            elif img_type == IMAGE_TYPE_OPTICAL_FLOW:
                # todo: fix me, optical flow buf should be 2D integers, however 1-D now
                opt = self.optical_flow_buf[render_idx].to(torch.float).detach().cpu().numpy()
                opt = self.width * (opt / 2 ** 15)
                opt = cv2.cvtColor(opt, cv2.COLOR_GRAY2BGR)
                render_img[:, 3 * self.width:] = opt
            else:
                raise NotImplementedError

        render_img = cv2.resize(render_img, dsize)
        cv2.imshow(f'{self.name}', render_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(0)
