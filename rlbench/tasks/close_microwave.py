from typing import List, Tuple
import numpy as np
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
import math


class CloseMicrowave(Task):

    def init_task(self) -> None:
        self.register_success_conditions([JointCondition(
            Joint('microwave_door_joint'), np.deg2rad(40))])
        self.door = Shape('microwave_door')
        self.microwave_door_joint = Joint('microwave_door_joint')

    def init_episode(self, index: int) -> List[str]:
        return ['close microwave',
                'shut the microwave',
                'close the microwave door',
                'push the microwave door shut']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 8.], [0, 0, np.pi / 8.]

    def boundary_root(self) -> Object:
        return Shape('boundary_root')
    
    def reward(self) -> float:
        arm_door_dist = np.linalg.norm(
            self.door.get_position() - self.robot.arm.get_tip().get_position())
        reach_door_reward = -arm_door_dist
        door_close_reward = -self.microwave_door_joint.get_joint_position()
        return door_close_reward + reach_door_reward
