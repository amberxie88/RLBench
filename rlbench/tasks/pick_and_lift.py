from typing import List
import numpy as np
from scipy.spatial.transform import Rotation
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors


class PickAndLift(Task):
    def init_task(self) -> None:
        self.target_block = Shape("pick_and_lift_target")
        self.distractors = [Shape("stack_blocks_distractor%d" % i) for i in range(2)]
        self.register_graspable_objects([self.target_block])
        self.boundary = SpawnBoundary([Shape("pick_and_lift_boundary")])
        self.success_detector = ProximitySensor("pick_and_lift_success")

        self._grasped_cond = GraspedCondition(self.robot.gripper, self.target_block)
        self._detected_cond = DetectedCondition(
            self.target_block, self.success_detector
        )

        cond_set = ConditionSet([self._grasped_cond, self._detected_cond])
        self.register_success_conditions([cond_set])

    def init_episode(self, index: int) -> List[str]:

        block_color_name, block_rgb = colors[index]
        self.target_block.set_color(block_rgb)

        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2,
            replace=False,
        )
        for i, ob in enumerate(self.distractors):
            name, rgb = colors[color_choices[int(i)]]
            ob.set_color(rgb)

        self.boundary.clear()
        self.boundary.sample(
            self.success_detector,
            min_rotation=(0.0, 0.0, 0.0),
            max_rotation=(0.0, 0.0, 0.0),
        )
        for block in [self.target_block] + self.distractors:
            self.boundary.sample(block, min_distance=0.1)

        return [
            "pick up the %s block and lift it up to the target" % block_color_name,
            "grasp the %s block to the target" % block_color_name,
            "lift the %s block up to the target" % block_color_name,
        ]

    def variation_count(self) -> int:
        return len(colors)

    def change_reward(self, task) -> None:
        self.reward_lang = task

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]

        if not grasped:
            grasp_block_reward = np.exp(
                -np.linalg.norm(
                    self.target_block.get_position() - self.robot.arm.get_tip().get_position()
                )
            )
            reward = grasp_block_reward
        else:
            lift_block_reward = np.exp(
                -np.linalg.norm(
                    self.target_block.get_position() - self.success_detector.get_position()
                )
            )
            reward = 1.0 + lift_block_reward
        
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
