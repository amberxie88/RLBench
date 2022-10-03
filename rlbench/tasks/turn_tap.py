from typing import List
import numpy as np
import math
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition

OPTIONS = ['left', 'right']


class TurnTap(Task):

    def init_task(self) -> None:
        self.left_start = Dummy('waypoint0')
        self.left_end = Dummy('waypoint1')
        self.right_start = Dummy('waypoint5')
        self.right_end = Dummy('waypoint6')
        self.left_joint = Joint('left_joint')
        self.right_joint = Joint('right_joint')

    def init_episode(self, index: int) -> List[str]:
        self.option = OPTIONS[index]
        if self.option == 'right':
            self.left_start.set_position(self.right_start.get_position())
            self.left_start.set_orientation(self.right_start.get_orientation())
            self.left_end.set_position(self.right_end.get_position())
            self.left_end.set_orientation(self.right_end.get_orientation())
            self.register_success_conditions(
                [JointCondition(self.right_joint, 1.57)])
            self.original_handle_position =  self.right_joint.get_joint_position()
        else:
            self.register_success_conditions(
                [JointCondition(self.left_joint, 1.57)])
            self.original_handle_position = self.left_joint.get_joint_position()

        return ['turn %s tap' % self.option,
                'rotate the %s tap' % self.option,
                'grasp the %s tap and turn it' % self.option]

    def variation_count(self) -> int:
        return 2

    def get_low_dim_state(self) -> np.ndarray:
        return np.concatenate(([self.right_joint.get_joint_position()], self.right_joint.get_position(),
                            [self.left_joint.get_joint_position()], self.right_joint.get_position()))

    def reward(self) -> float:
        if self.option == 'right':
            handle = self.right_joint
        else:
            handle = self.left_joint
        reach_handle_reward = -np.linalg.norm(
            handle.get_position() - self.robot.arm.get_tip().get_position())
        turn_handle_reward = math.fabs(handle.get_joint_position() - self.original_handle_position)
        return reach_handle_reward + turn_handle_reward