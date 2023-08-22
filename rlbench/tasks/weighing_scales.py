from typing import List, Tuple
import copy
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    DetectedCondition,
    NothingGrasped,
    GraspedCondition,
)


from rlbench.backend.spawn_boundary import SpawnBoundary

UNIQUE_PEPPERS_TO_PLACE = 3


class WeighingScales(Task):
    def init_task(self) -> None:
        self.needle = Shape("scales_meter_needle")
        self.needle_pivot = Shape("scales_meter_pivot")
        self.top_plate = Shape("scales_tray_visual")
        self.joint = Joint("scales_joint")

        _, _, starting_z = self.top_plate.get_position()
        self.top_plate_starting_z = starting_z
        self.needle_starting_ori = self.needle.get_orientation(
            relative_to=self.needle_pivot
        )

        self.peppers = [Shape("pepper%d" % i) for i in range(3)]
        self.register_graspable_objects(self.peppers)

        self.boundary = Shape("peppers_boundary")

        self.success_detector = ProximitySensor("success_detector")
        self.needle_detector = ProximitySensor("needle_sensor")
        self.success_conditions = [
            NothingGrasped(self.robot.gripper),
            DetectedCondition(self.needle, self.needle_detector),
        ]

        self.w0 = Dummy("waypoint0")
        self.w0_rel_pos = [
            +2.6203 * 10 ** (-3),
            -1.7881 * 10 ** (-7),
            +1.5197 * 10 ** (-1),
        ]
        self.w0_rel_ori = [-3.1416, -1.7467 * 10 ** (-2), -3.1416]

    def init_episode(self, index: int) -> List[str]:
        self.target_pepper_index = index
        while len(self.success_conditions) > 2:
            self.success_conditions.pop()
        self._grasped_cond = GraspedCondition(
            self.robot.gripper, self.peppers[self.target_pepper_index]
        )
        self._detected_cond = DetectedCondition(
            self.peppers[self.target_pepper_index], self.success_detector
        )
        self.success_conditions.append(self._detected_cond)
        self.register_success_conditions(self.success_conditions)
        b = SpawnBoundary([self.boundary])
        for p in self.peppers:
            b.sample(
                p,
                ignore_collisions=False,
                min_distance=0.12,
                min_rotation=(0.00, 0.00, -3.14),
                max_rotation=(0.00, 0.00, +3.14),
            )

        self._plate_pos = copy.deepcopy(self.top_plate.get_position())
        self._plate_above_pos = copy.deepcopy(self.top_plate.get_position())
        self._plate_above_pos[-1] = self.top_plate_starting_z + 0.25
        self._pepper_above_pos = copy.deepcopy(
            self.peppers[self.target_pepper_index].get_position()
        )
        self._pepper_above_pos[-1] = self.top_plate_starting_z + 0.25
        self._lifted = False
        self._above_plate = False

        self.w0.set_position(
            self.w0_rel_pos,
            relative_to=self.peppers[self.target_pepper_index],
            reset_dynamics=False,
        )
        self.w0.set_orientation(
            self.w0_rel_ori,
            relative_to=self.peppers[self.target_pepper_index],
            reset_dynamics=False,
        )
        pepper = {0: "green", 1: "red", 2: "yellow"}

        return [
            "weigh the %s pepper" % pepper[index],
            "pick up the %s pepper and set it down on the weighing scales"
            % pepper[index],
            "put the %s pepper on the scales" % pepper[index],
            "place the %s pepper onto the scales tray" % pepper[index],
            "lift up the %s pepper onto the weighing scales" % pepper[index],
            "grasp the %s pepper from above and lower it onto the tray" % pepper[index],
        ]

    def variation_count(self) -> int:
        return UNIQUE_PEPPERS_TO_PLACE

    def step(self):
        _, _, pos_z = self.top_plate.get_position()
        dz = self.top_plate_starting_z - pos_z
        d_alpha = -120 * dz
        new_needle_ori = [
            self.needle_starting_ori[0] + d_alpha,
            self.needle_starting_ori[1],
            self.needle_starting_ori[2],
        ]

        self.needle.set_orientation(
            new_needle_ori, relative_to=self.needle_pivot, reset_dynamics=False
        )

    def reward(self) -> float:
        pepper = self.peppers[self.target_pepper_index]
        grasped = self._grasped_cond.condition_met()[0]

        if not grasped:
            if self._detected_cond.condition_met()[0]:
                above_plate_reward = np.exp(
                    -np.linalg.norm(
                        self.robot.arm.get_tip().get_position() - self._plate_above_pos
                    )
                )
                reward = 5.0 + above_plate_reward
            else:
                grasp_reward = np.exp(
                    -np.linalg.norm(
                        self.robot.arm.get_tip().get_position() - pepper.get_position()
                    )
                )
                reward = grasp_reward
        else:
            grasp_reward = 1.0

            lift_reward = np.exp(
                -np.linalg.norm(
                    self.robot.arm.get_tip().get_position() - self._pepper_above_pos
                )
            )

            if not self._lifted and lift_reward > 0.9:
                self._lifted = True

            if self._lifted:
                lift_reward = 1.0

                above_plate_reward = np.exp(
                    -np.linalg.norm(
                        self.robot.arm.get_tip().get_position() - self._plate_above_pos
                    )
                )

                if not self._above_plate and above_plate_reward > 0.9:
                    self._above_plate = True

                if self._above_plate:
                    above_plate_reward = 1.0

                    reach_plate_reward = np.exp(
                        -np.linalg.norm(
                            self.robot.arm.get_tip().get_position() - self._plate_pos
                        )
                    )
                else:
                    reach_plate_reward = 0.0

            reward = (
                grasp_reward + lift_reward + above_plate_reward + reach_plate_reward
            )

        return reward

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -0.25 * np.pi], [0.0, 0.0, +0.25 * np.pi]

    def get_low_dim_state(self) -> np.ndarray:
        # For ad-hoc reward computation, attach reward
        reward = self.reward()
        state = super().get_low_dim_state()
        return np.hstack([reward, state])
