from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *

from shifu.units import Box, CameraSensor
from shifu.runner import run_policy, latest_logdir

import argparse
import torch

from examples.abb_pushbox_vision.task_config import (
    AbbRobotConfig,
    TableConfig,
    PushBoxConfig,
    GoalBoxConfig,
    PushBoxCameraConfig,
    VisionStageEnvConfig,
    VisionStagePPOConfig
)

from examples.abb_pushbox_vision.a_prior_stage import (
    LOG_ROOT,
    AbbRobot,
    AbbPushBox,
    RandPosBox,
    GoalBox
)

from examples.abb_pushbox_vision.b_regression_stage import get_multi_regressor


class FullVisionAbbPushBox(AbbPushBox):
    def __init__(self, cfg):
        super(AbbPushBox, self).__init__(cfg)
        self.robot = AbbRobot(AbbRobotConfig())
        self.table = Box(TableConfig())
        self.cube = RandPosBox(PushBoxConfig())
        self.goal = GoalBox(GoalBoxConfig())

        self.camera = CameraSensor(PushBoxCameraConfig())

        self.isg_env.create_envs(
            robot=self.robot,
            objects=[self.table, self.cube, self.goal],
            sensors=[self.camera]
        )

        # buffers
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # load regressor
        self.regressor = get_multi_regressor()
        encoder_logdir = latest_logdir(log_root=f"{LOG_ROOT}/Regression", run_name='MultiAE-l32')
        self.regressor.load(encoder_logdir)

    def compute_observations(self):
        prediction_dict = self.regressor({
            'rgb': self.camera.color_buf.permute(0, 3, 1, 2),
            'depth': self.camera.depth_buf.unsqueeze(3).permute(0, 3, 1, 2)
        })

        self.obs_buf = torch.cat([
            prediction_dict['obj_pos'].detach(),
            prediction_dict['goal_pos'].detach(),
            prediction_dict['ee_pos'].detach(),
        ], dim=1)


def get_args():
    parser = argparse.ArgumentParser("Abb Robot push box task")
    parser.add_argument("--run-mode",
                        '-r',
                        type=str,
                        choices=['train', 'play', 'random'],
                        help="run mode for push box task")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    run_args = get_args()
    ppo_config = VisionStagePPOConfig()
    ppo_config.runner.run_name = "VisionPushBox"
    run_policy(
        run_mode=run_args.run_mode,
        env_class=FullVisionAbbPushBox,
        env_cfg=VisionStageEnvConfig(),
        policy_cfg=ppo_config,
        log_root=f"{LOG_ROOT}/Vision",
    )
