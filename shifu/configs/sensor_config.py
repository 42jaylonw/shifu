from .base_config import BaseConfig
from isaacgym import gymapi


class BaseSensorConfig(BaseConfig):
    name = "DummySensor"
    frequency = 30  # [Hz]  # todo: add frequency sensor
    data_shape = 0


class CameraSensorConfig(BaseSensorConfig):
    name = "DummyCameraSensor"
    image_types = [
        gymapi.IMAGE_COLOR,
        gymapi.IMAGE_DEPTH,
        gymapi.IMAGE_SEGMENTATION,
        gymapi.IMAGE_OPTICAL_FLOW,
    ]
    image_normalization = False

    local_lookat_positions = None
    transform = None
    attach_local_transform = None   # not support yet

    def __init__(self):
        self._init_camera_props()
        super(CameraSensorConfig, self).__init__()

    def _init_camera_props(self):
        camera_props = gymapi.CameraProperties()

        for attr in dir(self.camera_props):
            if '__' not in attr:
                setattr(camera_props, attr, eval(f'self.camera_props.{attr}'))

        # change the attribute type to isaac tasks camera_props
        self.camera_props = camera_props

    class camera_props:
        enable_tensors = True
        use_collision_geometry = False
        width = 256
        height = 256
        near_plane = 0.1  # [m]
        far_plane = 3  # [m]
        horizontal_fov = 87  # Vertical field of view is calculated from height to width ratio
