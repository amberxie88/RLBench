from typing import List, Tuple
import numpy as np
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.task import Task
from pyrep.objects.joint import Joint



class OpenOven(Task):

    def init_task(self) -> None:
        self.door = Shape('oven_door')
        self.door_joint = Joint('oven_door_joint')
        self.register_success_conditions(
            [DetectedCondition(Shape('oven_door'), ProximitySensor('success')),
             NothingGrasped(self.robot.gripper)])

    def init_episode(self, index: int) -> List[str]:
        return ['open the oven',
                'open the oven door',
                'grab hold of the the handle and pull the oven door open']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 4.], [0, 0, np.pi / 4.]

    def boundary_root(self) -> Object:
        return Shape('oven_boundary_root')
    
    def reward(self) -> float:
        arm_door_dist = np.linalg.norm(
            self.door.get_position() - self.robot.arm.get_tip().get_position())
        reach_door_reward = -arm_door_dist
        door_close_reward = -self.door_joint.get_joint_position()
        return reach_door_reward + door_close_reward
