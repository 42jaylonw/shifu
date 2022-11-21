import numpy as np
import pybullet as p


class PybBox:
    def __init__(self,
                 box_dim,
                 base_pos,
                 base_quat,
                 fixed_base,
                 mass,
                 color: list,
                 pybullet_client):
        self.pybullet_client = pybullet_client
        if fixed_base:
            mass = 0
        if len(color) == 3:
            color.append(1.)

        self.v_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=np.array(box_dim) / 2,
            rgbaColor=color,
            physicsClientId=self.pybullet_client,
        )
        self.c_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=np.array(box_dim) / 2,
            physicsClientId=self.pybullet_client,
        )
        self.body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=self.c_id,
            baseVisualShapeIndex=self.v_id,
            basePosition=base_pos,
            baseOrientation=base_quat,
            physicsClientId=self.pybullet_client,
        )
