import copy
from typing import List
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import NothingGrasped, DetectedCondition, GraspedCondition
from rlbench.backend.task import Task

MEAT = ['chicken', 'steak']


class MeatOnGrill(Task):

    def init_task(self) -> None:
        self._steak = Shape('steak')
        self._chicken = Shape('chicken')
        self._success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self._chicken, self._steak])
        self._w1 = Dummy('waypoint1')
        self._w1z = self._w1.get_position()[2]

    def init_episode(self, index: int) -> List[str]:
        conditions = [NothingGrasped(self.robot.gripper)]
        if index == 0:
            self._target = self._chicken
        else:
            self._target = self._steak

        x, y, _ = self._target.get_position()
        self._w1.set_position([x, y, self._w1z])
        self._detected_cond = DetectedCondition(self._target, self._success_sensor)
        conditions.append(self._detected_cond)

        self._grasped_cond = GraspedCondition(self.robot.gripper, self._target)
        self.target1_pos = copy.deepcopy(self._w1.get_position())
        self.target2_pos = copy.deepcopy(self._success_sensor.get_position())
        self.target2_pos[-1] = self._w1z
        self.lifted = False
        self.above_grill = False

        self.register_success_conditions(conditions)
        return ['put the %s on the grill' % MEAT[index],
                'pick up the %s and place it on the grill' % MEAT[index],
                'grill the %s' % MEAT[index]]

    def variation_count(self) -> int:
        return 2

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]

        if not grasped:
            if self._detected_cond.condition_met()[0]:
                reward = 5.0
            else:
                grasp_target_reward = np.exp(
                    -np.linalg.norm(
                        self.robot.arm.get_tip().get_position()
                        - self._target.get_position()
                    )
                )
                reward = grasp_target_reward
        else:
            lift_target_reward = np.exp(
                -np.linalg.norm(
                    self.robot.arm.get_tip().get_position()
                    - self.target1_pos)
            )

            # TODO: Add more formal condition for reaching
            if not self.lifted and lift_target_reward > 0.9:
                self.lifted = True

            if self.lifted:
                lift_target_reward = 1.0

                above_grill_reward = np.exp(
                    -np.linalg.norm(
                        self.robot.arm.get_tip().get_position()
                        - self.target2_pos)
                )

                if not self.above_grill and above_grill_reward > 0.9:
                    self.above_grill = True

                if not self.above_grill:
                    reach_grill_reward = 0.0
                else:
                    above_grill_reward = 1.0
                    reach_grill_reward = np.exp(
                        -np.linalg.norm(
                            self.robot.arm.get_tip().get_position()
                            - self._success_sensor.get_position()
                        )
                    )
            else:
                above_grill_reward = 0.0
                reach_grill_reward = 0.0

            grasp_target_reward = 1.0

            reward = grasp_target_reward + lift_target_reward + above_grill_reward + reach_grill_reward

        return reward
    
    def get_low_dim_state(self) -> np.ndarray:
        # For ad-hoc reward computation, attach reward
        reward = self.reward()
        state = super().get_low_dim_state()
        return np.hstack([reward, state])