"""
Procedural objects supplied from:
https://sites.google.com/site/brainrobotdata/home/models
"""

from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

from rlbench.backend.conditions import ConditionSet, DetectedCondition, GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.task_utils import sample_procedural_objects
from rlbench.const import colors
import random

GOAL_COND = False
DENSE_REWARD = False
SPAWN_BOUNDARY_HYPOTENUSE = 0.3436538967 * 1.5

class PickFromContainer(Task):

    def init_task(self) -> None:
        self.large_container = Shape('large_container')
        self.success_detector = ProximitySensor('pick_and_lift_success')
        if GOAL_COND:
            self.goal_sensor = VisionSensor('goal_sensor')
            self.goal_sensor.set_resolution([64, 64])
            self.goal_container = Shape('goal_container')
            self.goal_image = None
        self.success_detector.set_position([.278466582, -.00815787166, 1.47197676])
        self.target_waypoint = Dummy('waypoint3')
        self.spawn_boundary_shape = Shape('spawn_boundary')
        self.spawn_boundary = SpawnBoundary([self.spawn_boundary_shape])
        self.register_waypoint_ability_start(1, self._move_above_object)
        self.register_waypoints_should_repeat(self._repeat)
        self.num_objects = 5

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index            
        self.bin_objects = sample_procedural_objects(self.get_base(), self.num_objects)
        self.register_graspable_objects(self.bin_objects)
        self.spawn_boundary.clear()
        value = random.random()
        self.large_container.set_color([value, value, value])
        self.large_container.set_position(
                [0.0, 0.0, 0.01], relative_to=self.large_container,
                reset_dynamics=True)
        if GOAL_COND:
            self.bin_objects[0].set_position(
                [0.0, 0.0, 0.2], relative_to=self.large_container,
                reset_dynamics=True)
            self.spawn_boundary.sample(
                self.bin_objects[0], ignore_collisions=True, min_distance=0.05)
        else:
            for ob in self.bin_objects:
                ob.set_position(
                    [0.0, 0.0, 0.2], relative_to=self.large_container,
                    reset_dynamics=True)
                self.spawn_boundary.sample(
                    ob, ignore_collisions=True, min_distance=0.05)
        target_pos = [-5.9605e-8, -2.5005e-1, +1.7e-1]
        self.target_waypoint.set_position(
            target_pos, relative_to=self.large_container, reset_dynamics=True)
        self.grasped_cond = GraspedCondition(self.robot.gripper, self.bin_objects[0])
        self.conditions = []
        for ob in self.bin_objects:
            self.conditions.append(GraspedCondition(self.robot.gripper, ob))
        self.register_success_conditions([ConditionSet([self.grasped_cond])])
        if GOAL_COND:
            self.goal_container.set_position([0.7, 0.0, 0.0], relative_to=self.goal_container,
                reset_dynamics=True)
            for ob in self.bin_objects[1:]:
                ob.set_position(
                    [0.0, 0.0, 0.2], relative_to=self.goal_container,
                    reset_dynamics=True)
            self.goal_sensor.set_position([0.0, 0.0, 0.0], relative_to=self.bin_objects[0],
                    reset_dynamics=False)
            self.goal_sensor.set_quaternion([0, 1, 0, 0])
        return ['pick and lift and object from the container']

    def variation_count(self) -> int:
        return len(colors)
    
    def set_num_objects(self, num_objects) -> None:
        self.num_objects = num_objects
    
    def spawn_remaining_objects(self) -> None:
        for ob in self.bin_objects[1:]:
            ob.set_position(
                [0.0, 0.0, 0.2], relative_to=self.large_container,
                reset_dynamics=True)
            self.spawn_boundary.sample(
                ob, ignore_collisions=True, min_distance=0.05)

    def any_object_is_grasped(self) -> bool:
        for cond in self.conditions: 
            if cond.condition_met()[0]:
                return True
        return False

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def reward(self) -> float:
        if DENSE_REWARD:
            dist = np.linalg.norm(self.bin_objects[0].get_position() - self.robot.arm.get_tip().get_position())
            reward = np.clip(1 - (dist / SPAWN_BOUNDARY_HYPOTENUSE), 0, 1)
        else:
            reward = 0.0
        return reward

    def cleanup(self) -> None:
        [ob.remove() for ob in self.bin_objects if ob.still_exists()]
        self.bin_objects = []

    def step(self) -> None:
        pass
           
    def _move_above_object(self, waypoint):
        if len(self.bin_objects_not_done) <= 0:
            raise RuntimeError('Should not be here.')
        bin_obj = self.bin_objects_not_done[0]
        way_obj = waypoint.get_waypoint_object()
        way_obj.set_position(bin_obj.get_position())
        x, y, _ = way_obj.get_orientation()
        _, _, z = bin_obj.get_orientation(relative_to=way_obj)
        way_obj.set_orientation([x, y, z])

    def _repeat(self):
        return len(self.bin_objects_not_done) > 0
