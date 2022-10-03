from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
import numpy as np


class SlideBlockToTarget(Task):

    def init_task(self) -> None:
        self._block = Shape('block')
        self._target = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(self._block, self._target)])

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        return ['slide the block to target',
                'slide the block onto the target',
                'push the block until it is sitting on top of the target',
                'slide the block towards the green target',
                'cover the target with the block by pushing the block in its'
                ' direction']

    def variation_count(self) -> int:
        return 1
    
    def reward(self) -> float:
        reach_reward = -np.linalg.norm(
            self._block.get_position() - self.robot.arm.get_tip().get_position())
        push_reward = -np.linalg.norm(
            self._target.get_position() - self._block.get_position())
        return reach_reward + push_reward
