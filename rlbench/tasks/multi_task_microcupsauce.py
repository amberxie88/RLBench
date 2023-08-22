from typing import List, Tuple
import numpy as np
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import JointCondition
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition, ConditionSet, GraspedCondition
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary

class MultiTaskMicrocupsauce(Task):

    def init_task(self) -> None:
        self.multitask_env = True
        self.reward_lang = None
        # saucepan
        self.lid = Shape("saucepan_lid_grasp_point")
        self.sauce_success_detector = ProximitySensor("success0")
        self.register_graspable_objects([self.lid])
        self._sauce_grasped_cond = GraspedCondition(self.robot.gripper, self.lid)
        self._sauce_detected_cond = DetectedCondition(self.lid, self.sauce_success_detector)
        # microwave
        self.door = Shape('microwave_door')
        self.microwave_door_joint = Joint('microwave_door_joint')
        # cup
        self.cup1 = Shape("cup1")
        # self.cup2 = Shape("cup2")
        self.cup1_visual = Shape("cup1_visual")
        # self.cup2_visual = Shape("cup2_visual")
        self.boundary = SpawnBoundary([Shape("boundary")])
        self.success_sensor = ProximitySensor("success1")
        self.register_graspable_objects([self.cup1])

        self._cup_grasped_cond = GraspedCondition(self.robot.gripper, self.cup1)
        self._cup_detected_cond = DetectedCondition(self.cup1, self.success_sensor, negated=True)


        sauce_cond_set = ConditionSet([self._sauce_grasped_cond, self._sauce_detected_cond])
        cup_cond_set = ConditionSet([self._cup_grasped_cond, self._cup_detected_cond])
        # Note: Impossible to grasp cup and saucepan at the same time
        self.register_success_conditions(
                [JointCondition(Joint('microwave_door_joint'), np.deg2rad(37)), 
                cup_cond_set,
                sauce_cond_set
             ]) 

    def change_reward(self, task) -> None:
        self.reward_lang = task

    def reward(self) -> float:
        if self.reward_lang == "take_lid_off_saucepan":
            grasped = self._sauce_grasped_cond.condition_met()[0]
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
                        self.lid.get_position() - self.sauce_success_detector.get_position()
                    )
                )
                reward = 1.0 + lift_lid_reward
            return reward
        elif self.reward_lang == "close_microwave":
            door_close_reward = -self.microwave_door_joint.get_joint_position()
            return door_close_reward
        elif self.reward_lang == "pick_up_cup":
            grasped = self._cup_grasped_cond.condition_met()[0]

            if not grasped:
                grasp_cup1_reward = np.exp(
                    -np.linalg.norm(
                        self.cup1.get_position() - self.robot.arm.get_tip().get_position()
                    )
                )
                reward = grasp_cup1_reward
            else:
                lift_cup1_reward = np.exp(
                    -np.linalg.norm(
                        self.cup1.get_position() - self.success_sensor.get_position()
                    )
                )
                reward = 1.0 + lift_cup1_reward
            return reward
        else:
            raise ValueError("Invalid task reward specified")

    def init_episode(self, index: int) -> List[str]:
        # can change to include all possible tasks (microwave, tap, fridge)
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
