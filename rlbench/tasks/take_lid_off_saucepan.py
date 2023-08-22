from typing import List

import numpy as np
from scipy.spatial.transform import Rotation
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, ConditionSet, GraspedCondition
from rlbench.backend.task import Task


class TakeLidOffSaucepan(Task):
    def init_task(self) -> None:
        self.lid = Shape("saucepan_lid_grasp_point")
        self.success_detector = ProximitySensor("success")
        self.register_graspable_objects([self.lid])
        self._grasped_cond = GraspedCondition(self.robot.gripper, self.lid)
        self._detected_cond = DetectedCondition(self.lid, self.success_detector)

        cond_set = ConditionSet([self._grasped_cond, self._detected_cond])
        self.register_success_conditions([cond_set])

    def init_episode(self, index: int) -> List[str]:
        return [
            "take lid off the saucepan",
            "using the handle, lift the lid off of the pan",
            "remove the lid from the pan",
            "grip the saucepan's lid and remove it from the pan",
            "leave the pan open",
            "uncover the saucepan",
        ]

    def variation_count(self) -> int:
        return 1

    def change_reward(self, task) -> None:
        self.reward_lang = task

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]
        if not grasped:
            grasp_lid_reward = np.exp(
                -np.linalg.norm(
                    self.lid.get_position() - self.robot.arm.get_tip().get_position()
                )
            )
            reward = grasp_lid_reward
        else:
            lift_lid_reward = np.exp(
                -np.linalg.norm(
                    self.lid.get_position() - self.success_detector.get_position()
                )
            )
            reward = 1.0 + lift_lid_reward

        # # Vertical gripper reward
        # curr_quat = self.robot.gripper.get_quaternion()
        # goal_quat = [1, 0, 0, 0]
        # euler = Rotation.from_quat(curr_quat).as_euler("xyz", degrees=True) + 180
        # goal_euler = Rotation.from_quat(goal_quat).as_euler("xyz", degrees=True) + 180
        # euler_dist = [
        #     min(abs(e - ge), 360 - abs(e - ge)) / 360.0
        #     for e, ge in zip(euler, goal_euler)
        # ]
        # gripper_reward = np.exp(-np.linalg.norm(euler_dist))

        # reward += gripper_reward
        return reward

    def get_low_dim_state(self) -> np.ndarray:
        # For ad-hoc reward computation, attach reward
        reward = self.reward()
        state = super().get_low_dim_state()
        return np.hstack([reward, state])
