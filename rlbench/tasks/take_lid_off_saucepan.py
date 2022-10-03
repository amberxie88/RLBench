from typing import List

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.task import Task


class TakeLidOffSaucepan(Task):

    def init_task(self) -> None:
        self.lid = Shape('saucepan_lid_grasp_point')
        self.success_detector = ProximitySensor('success')
        self.register_graspable_objects([self.lid])
        self.grasped_condition = GraspedCondition(self.robot.gripper, self.lid)
        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.lid),
            DetectedCondition(self.lid, self.success_detector)
        ])
        self.register_success_conditions([cond_set])

    def init_episode(self, index: int) -> List[str]:
        return ['take lid off the saucepan',
                'using the handle, lift the lid off of the pan',
                'remove the lid from the pan',
                'grip the saucepan\'s lid and remove it from the pan',
                'leave the pan open',
                'uncover the saucepan']

    def variation_count(self) -> int:
        return 1

    # def reward(self) -> float:
    #     grasp_lid_reward = -np.linalg.norm(
    #         self.lid.get_position() - self.robot.arm.get_tip().get_position())
    #     lift_lid_reward = -np.linalg.norm(
    #         self.lid.get_position() - self.success_detector.get_position())
    #     grasped_reward, _ = self.grasped_condition.condition_met()
    #     return grasp_lid_reward + lift_lid_reward + int(grasped_reward)
    
    def reward(self) -> float:
        grasped = self.grasped_condition.condition_met()[0]
        
        if not grasped:
            grasp_lid_reward = np.exp(-np.linalg.norm(
                self.lid.get_position() - self.robot.arm.get_tip().get_position()))
            reward = grasp_lid_reward
        else:
            lift_lid_reward = np.exp(-np.linalg.norm(
                self.lid.get_position() - self.success_detector.get_position()))
            reward = 10. + lift_lid_reward
        return reward
