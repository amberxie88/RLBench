from typing import List, Tuple
import numpy as np
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task


class MultiTaskMicrofridgetap(Task):

    def init_task(self) -> None:
        self.multitask_env = True
        self.right_joint = Joint('right_joint')
        self.register_success_conditions(
                [JointCondition(
            Joint('microwave_door_joint'), np.deg2rad(37)), JointCondition(Joint('top_joint'), np.deg2rad(70)),
             JointCondition(self.right_joint, 1.55)]) 

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
