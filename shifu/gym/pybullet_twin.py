import time

from typing import List, Dict
import pybullet as p
import pybullet_data as pd
import numpy as np
import matplotlib.pyplot as plt

from shifu.configs import (BoxActorConfig, CameraSensorConfig, ArmRobotActorConfig)
from shifu.utils.pyb_utils import PybBox
import torch
from shifu.utils.torch_utils import inverse_kinematics


# todo: [add more types of robot]


class PybSensor(object):
    def _init_buf(self):
        raise NotImplementedError

    def refresh_state(self):
        raise NotImplementedError


class PybulletTwinEnv:
    sensor_cfg_list: Dict[str, PybSensor]

    def __init__(
            self,
            dt,
            robot_cfg,
            obj_cfg_list=(),
            sensor_cfg_list=(),
            decimation=10,
            render_mode=1,
            video_recording_filename='data/test.mp4'
    ):
        self.dt = dt
        self.robot_cfg = robot_cfg
        self.obj_cfg_list = obj_cfg_list
        self.sensor_cfg_list = sensor_cfg_list
        self.decimation = decimation

        self._render_mode = render_mode
        if self._render_mode > 0:
            self.pybullet_client = p.connect(
                p.GUI,
                options=f"--width=1280 --height=720 --mp4fps=60 --mp4=\"{video_recording_filename}\""
                if self._render_mode == 2
                else "--width=1280 --height=720 --mp4fps=60"
            )
        else:
            self.pybullet_client = p.connect(p.DIRECT)

        self._build_env()

    def _build_env(self):
        p.resetSimulation()

        if self._render_mode > 0:
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # p.setRealTimeSimulation(1)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dt / self.decimation)
        p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=-20,
            cameraPitch=-40,
            cameraTargetPosition=(0, 0, 0))

        p.setAdditionalSearchPath(pd.getDataPath())
        p.loadURDF('plane_implicit.urdf')

        # get robot and objects
        urdf_filename = self.robot_cfg.urdf_filename.replace('_isaac', '')
        try:
            self.robot_body_id = p.loadURDF(f"{self.robot_cfg.root_dir}/{urdf_filename}",
                                            basePosition=self.robot_cfg.default_pos,
                                            useFixedBase=self.robot_cfg.asset_options.fix_base_link)
        except:
            raise NotImplementedError(
                "please provide a non-isaac urdf with name xxx.urdf and isaac urdf as xxx_isaac.urdf")
        self.default_joint_pos = self.robot_cfg.default_dof_pos

        # get joint Info
        self.joint_ids = []
        self.link_name_id_dict = {}
        self.joint_info = {}
        for j in range(p.getNumJoints(self.robot_body_id)):
            p.changeDynamics(self.robot_body_id, j, linearDamping=0, angularDamping=0)
            joint_info = p.getJointInfo(self.robot_body_id, j)
            joint_name = joint_info[1]
            joint_type = joint_info[2]
            link_name = joint_info[12].decode()
            self.link_name_id_dict[link_name] = j
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                self.joint_ids.append(j)
                self.joint_info[joint_name] = joint_info

        # load sensors
        self.sensor_dict = {}
        for sensor_cfg in self.sensor_cfg_list:
            if isinstance(sensor_cfg, CameraSensorConfig):
                self.sensor_dict[sensor_cfg.name] = PybCamera(
                    width=sensor_cfg.camera_props.width,
                    height=sensor_cfg.camera_props.height,
                    camera_position=sensor_cfg.local_pos,
                    camera_target_position=sensor_cfg.target_pos,
                    near_plane=sensor_cfg.camera_props.near_plane,
                    far_plane=sensor_cfg.camera_props.far_plane,
                    field_of_view=sensor_cfg.camera_props.horizontal_fov,
                    pybullet_id=self.pybullet_client,
                )
        # load objects in the scenario
        self.object_body_id_dict = {}
        for obj_cfg in self.obj_cfg_list:
            if isinstance(obj_cfg, BoxActorConfig):
                obj_idx = PybBox(box_dim=obj_cfg.box_dim,
                                 base_pos=obj_cfg.default_pos,
                                 base_quat=obj_cfg.default_quat,
                                 mass=obj_cfg.mass,
                                 color=obj_cfg.color,
                                 fixed_base=obj_cfg.asset_options.fix_base_link,
                                 pybullet_client=self.pybullet_client).body_id
            else:
                obj_idx = p.loadURDF(f"{obj_cfg.root_dir}/{obj_cfg.urdf_filename}",
                                     basePosition=obj_cfg.default_pos,
                                     useFixedBase=obj_cfg.asset_options.fix_base_link)
            self.object_body_id_dict[obj_cfg.name] = obj_idx

    def jog_joints(self):
        action_selector_ids = []
        for joint_name, joint_info in self.joint_info.items():
            joint_high = joint_info[9]
            joint_low = joint_info[8]
            action_selector_id = p.addUserDebugParameter(paramName=joint_name.decode(),
                                                         rangeMin=joint_low,
                                                         rangeMax=joint_high,
                                                         startValue=0)
            action_selector_ids.append(action_selector_id)

        while True:
            joint_positions = np.zeros(len(self.joint_ids))
            for dim in range(len(self.joint_ids)):
                joint_positions[dim] = p.readUserDebugParameter(action_selector_ids[dim])

            p.setJointMotorControlArray(self.robot_body_id, self.joint_ids, p.POSITION_CONTROL, joint_positions)
            p.stepSimulation()

            for sensor in self.sensor_dict.values():
                sensor.refresh_state()

    def reset(self):
        for i, j_id in enumerate(self.joint_ids):
            p.resetJointState(self.robot_body_id, j_id, self.default_joint_pos[i])
        for sensor in self.sensor_dict.values():
            sensor.refresh_state()

    def step(self, action):
        raise NotImplementedError("Override me to define your own action space")

    def joint_command(self, target_joint_position):
        _itp = np.subtract(target_joint_position, self.get_joint_position()) / self.decimation
        _start_joint_position = self.get_joint_position()
        for i in range(1, self.decimation + 1):
            step_target_joint_position = _start_joint_position + i * _itp
            self._apply_joint_positions(step_target_joint_position)
        for sensor in self.sensor_dict.values():
            sensor.refresh_state()

    def _apply_joint_positions(self, joint_positions):
        init_pos = self.get_joint_position()
        interp_slice = np.subtract(joint_positions, init_pos) / self.decimation
        for i in range(1, self.decimation + 1):
            step_tar_joi_pos = init_pos + interp_slice * i
            p.setJointMotorControlArray(self.robot_body_id, self.joint_ids, p.POSITION_CONTROL, step_tar_joi_pos)
            p.stepSimulation()
            if self._render_mode > 0:
                time.sleep(self.dt / self.decimation)

    def get_joint_position(self):
        return np.array([p.getJointState(self.robot_body_id, x)[0] for x in self.joint_ids])

    def get_joint_velocity(self):
        return np.array([p.getJointState(self.robot_body_id, x)[1] for x in self.joint_ids])

    def get_robot_root_pose(self):
        return p.getBasePositionAndOrientation(self.robot_body_id)

    def get_object_root_pose(self, body_name):
        return p.getBasePositionAndOrientation(self.object_body_id_dict[body_name])


class ArmTwin(PybulletTwinEnv):
    robot_config: ArmRobotActorConfig

    def _build_env(self):
        super(ArmTwin, self)._build_env()
        self.end_effector_id = self.link_name_id_dict[self.robot_cfg.end_effector_names[0]]
        self.end_effector_vel = self.robot_cfg.end_effector_velocity

    def get_end_effector_pose(self):
        ee_info = p.getLinkState(self.robot_body_id, self.end_effector_id)
        ee_pos = ee_info[0]
        ee_quat = ee_info[1]
        return ee_pos, ee_quat

    def step(self, action):
        ee_pos, _ = self.get_end_effector_pose()
        tar_ee_pose = ee_pos + action * self.dt * self.end_effector_vel
        self.target_ee_pos_command(tar_ee_pose)

    def target_ee_pos_command(self, tar_pos, tar_quat=None, max_iter=100):
        ee_pos, _ = self.get_end_effector_pose()
        _itp_pos = np.subtract(tar_pos, ee_pos) / self.decimation
        for i in range(1, self.decimation + 1):
            step_ee_pos = ee_pos + i * _itp_pos
            target_joint_position = p.calculateInverseKinematics(
                self.robot_body_id,
                self.end_effector_id,
                targetPosition=step_ee_pos,
                targetOrientation=tar_quat,
                maxNumIterations=max_iter
            )
            self._apply_joint_positions(target_joint_position)
        for sensor in self.sensor_dict.values():
            sensor.refresh_state()

    def target_ee_pos_command_torch_ik(self, tar_pos, tar_quat=(0.7071787, 0.7071787, 0, 0), device='cpu'):
        ee_pos, ee_quat = self.get_end_effector_pose()
        _itp_pos = np.subtract(tar_pos, ee_pos) / self.decimation
        for i in range(1, self.decimation + 1):
            step_tar_ee_pos = ee_pos + i * _itp_pos

            # result = p.getLinkState(self.robot_body_id,
            #                         self.end_effector_id,
            #                         computeLinkVelocity=1,
            #                         computeForwardKinematics=1)
            # link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
            _zeros = [0] * len(self.get_joint_position())
            j_pos = [p.getJointState(self.robot_body_id, x)[0] for x in self.joint_ids]
            j_ee = p.calculateJacobian(
                bodyUniqueId=self.robot_body_id,
                linkIndex=self.end_effector_id,
                localPosition=(0., 0., 0.),
                objPositions=j_pos,
                objVelocities=_zeros,
                objAccelerations=_zeros)

            target_joint_position = inverse_kinematics(
                torch.tensor(self.get_joint_position(), dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(ee_pos, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(ee_quat, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(step_tar_ee_pos, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(tar_quat, dtype=torch.float32, device=device).unsqueeze(0),
                j_ee=torch.tensor(j_ee, dtype=torch.float32, device=device).view(6, -1).unsqueeze(0),
                damping=0.1,
                device=device
            ).detach().cpu().numpy().flatten()

            self._apply_joint_positions(target_joint_position)
        for sensor in self.sensor_dict.values():
            sensor.refresh_state()


def look_at(vec_pos, vec_look_at):
    z = vec_look_at - vec_pos
    z = z / np.linalg.norm(z)

    x = np.cross(z, np.array([0., 1., 0.]))
    x = x / np.linalg.norm(x)

    y = np.cross(x, z)
    y = y / np.linalg.norm(y)

    view_mat = np.zeros((3, 3))

    view_mat[:3, 0] = x
    view_mat[:3, 1] = y
    view_mat[:3, 2] = z

    return view_mat


def pyb_real_depth(depth_img, near, far):
    # pybullet depth = far * near / (far - (far - near) * depthImg) # depthImg is the depth from Bullet 'getCameraImage'
    depth = far * near / (far - (far - near) * depth_img)
    return depth


class PybCamera(PybSensor):
    rgb_buf: np.ndarray
    depth_buf: np.ndarray
    seg_buf: np.ndarray

    def __init__(
            self,
            width,
            height,
            camera_position,
            camera_target_position,
            near_plane,
            far_plane,
            field_of_view,
            pybullet_id,
            camera_orientation=None,
            light_direction=(0., 0., 0.),
            light_color=(1., 1., 1.),
            light_distance=1.,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
    ):
        self.width = width
        self.height = height
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.pybullet_id = pybullet_id
        self.renderer = renderer
        camera_distance = np.linalg.norm(np.subtract(camera_position, camera_target_position))

        if camera_orientation is not None:
            matrix = p.getMatrixFromQuaternion(camera_orientation, physicsClientId=self.pybullet_id)
            # tx_vec = np.array([matrix[0], matrix[3], matrix[6]])
            # ty_vec = np.array([matrix[1], matrix[4], matrix[7]])
            tz_vec = np.array([matrix[2], matrix[5], matrix[8]])

            self._view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_position,
                cameraTargetPosition=camera_target_position,
                cameraUpVector=tz_vec,
                physicsClientId=self.pybullet_id
            )
        else:
            # calculate view matrix using camera_position and camera_target_position
            diff = np.subtract(camera_target_position, camera_position)
            yaw = np.rad2deg(np.arctan2(diff[0], diff[1]))
            pitch = np.rad2deg(np.arctan2(diff[2], diff[1]))
            self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=camera_target_position,
                distance=camera_distance,
                yaw=yaw,
                pitch=pitch,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.pybullet_id
            )

        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=field_of_view,
            aspect=float(self.width) / self.height,
            nearVal=near_plane,
            farVal=far_plane,
            physicsClientId=self.pybullet_id
        )

        self.view_matrix = np.matrix(self._view_matrix).reshape(4, 4)
        self.proj_matrix = np.matrix(self._proj_matrix).reshape(4, 4)

        self.light_param = dict(lightDirection=light_direction, lightColor=light_color, lightDistance=light_distance)
        self._init_buf()

    def _init_buf(self):
        self.rgb_buf = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.depth_buf = np.zeros((self.width, self.height), dtype=np.float32)
        self.seg_buf = np.zeros((self.width, self.height), dtype=np.int32)

    def refresh_state(self):
        width, height, rgb_pixels, depth_pixels, segmentation_mask = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=self.renderer,
            **self.light_param
        )
        self.rgb_buf = np.float32(rgb_pixels[:, :, :3]) / 255
        # convert to real depth
        self.depth_buf = pyb_real_depth(depth_pixels, self.near_plane, self.far_plane)

        self.seg_buf = segmentation_mask


if __name__ == '__main__':
    import cv2
    from shifu.utils.camera import get_pixel_position, get_world_position
    from examples.abb_pushbox_vision.task_config import (
        AbbRobotAssetConfig, TableAssetConfig, PushBoxConfig, GoalBoxConfig, PerceptualCameraConfig
    )

    robot_config = AbbRobotAssetConfig()
    robot_config.end_effector_velocity = 0.3
    # robot_config.urdf_filename = "urdf/abb_rod_description/urdf/irb1200_7_70.urdf"
    obj_cfg = PushBoxConfig()
    obj_cfg.default_pos = [0.1, -0.7, 0.125]
    goal_cfg = GoalBoxConfig()
    goal_cfg.default_pos = [-0.05, -0.3, 0.04]

    arm = ArmTwin(
        dt=0.1,
        robot_cfg=robot_config,
        obj_cfg_list=[TableAssetConfig(), obj_cfg, goal_cfg],
        sensor_cfg_list=[PerceptualCameraConfig()],
    )
    # p_env.jog_joints()
    arm.reset()
    for t in range(1000):
        obj_pos = np.array(arm.get_object_root_pose('box')[0])
        goal_pos = np.array(arm.get_object_root_pose('goal')[0])
        ee_pos = np.array(arm.get_end_effector_pose()[0])

        width = arm.sensor_dict['rgbd_camera'].width
        height = arm.sensor_dict['rgbd_camera'].height
        view_matrix = arm.sensor_dict['rgbd_camera'].view_matrix
        proj_matrix = arm.sensor_dict['rgbd_camera'].proj_matrix

        obj_pixel_pos = get_pixel_position(obj_pos, view_matrix, proj_matrix, width, height)
        goal_pixel_pos = get_pixel_position(goal_pos, view_matrix, proj_matrix, width, height)
        ee_pixel_pos = get_pixel_position(ee_pos, view_matrix, proj_matrix, width, height)
        rgb = arm.sensor_dict['rgbd_camera'].rgb_buf

        # rgb[128 - obj_pixel_pos[1], obj_pixel_pos[0]] = np.zeros(3)
        # rgb[goal_pixel_pos[0], goal_pixel_pos[1]] = np.ones(3)
        mc = (0, 255, 0)
        cv2.drawMarker(rgb, (obj_pixel_pos[0], obj_pixel_pos[1]), mc, markerType=1, markerSize=5)
        cv2.drawMarker(rgb, (goal_pixel_pos[0], goal_pixel_pos[1]), mc, markerType=2, markerSize=5)
        cv2.drawMarker(rgb, (ee_pixel_pos[0], ee_pixel_pos[1]), mc, markerType=0, markerSize=5)

        render_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Show images
        cv2.namedWindow('Prediction', cv2.WINDOW_AUTOSIZE)
        _scale = 2
        render_img = cv2.resize(render_img, (render_img.shape[1] * _scale,
                                             render_img.shape[0] * _scale))
        cv2.imshow('Prediction', render_img)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()

        rand_ee_pos = np.random.uniform(-1, 1, 2)
        tar_ee_pose = np.concatenate([rand_ee_pos, [0]])
        arm.step(tar_ee_pose)
