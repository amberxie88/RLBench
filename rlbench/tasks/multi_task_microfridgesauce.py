from typing import List, Tuple
import numpy as np
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import JointCondition
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition, ConditionSet, GraspedCondition
from rlbench.backend.task import Task

class MultiTaskMicrofridgesauce(Task):

    def init_task(self) -> None:
        self.multitask_env = True
        self.lid = Shape("saucepan_lid_grasp_point")
        self.success_detector = ProximitySensor("success")
        self.register_graspable_objects([self.lid])
        self._grasped_cond = GraspedCondition(self.robot.gripper, self.lid)
        self._detected_cond = DetectedCondition(self.lid, self.success_detector)
        self.door = Shape('microwave_door')
        self.microwave_door_joint = Joint('microwave_door_joint')
        self.fridge_door_joint = Joint('top_joint')
        self.reward_lang = None

        # this success condition doesn't work, but demo looks good
        cond_set = ConditionSet([self._grasped_cond, self._detected_cond])
        self.register_success_conditions(
                [JointCondition(Joint('microwave_door_joint'), np.deg2rad(37)), 
                JointCondition(Joint('top_joint'), np.deg2rad(70)),
                cond_set
             ]) 
    
    def change_reward(self, task) -> None:
        self.reward_lang = task

    def reward(self) -> float:
        if self.reward_lang == "take_lid_off_saucepan":
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
            return reward
        elif self.reward_lang == "close_microwave":
            door_close_reward = -self.microwave_door_joint.get_joint_position()
            return door_close_reward
        elif self.reward_lang == "open_fridge":
            door_open_reward = -self.fridge_door_joint.get_joint_position()
            return door_open_reward
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
