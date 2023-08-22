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
# from rlbench.backend.shapenet_utils import ShapenetSampler
from rlbench.const import colors
import random

DENSE_REWARD = False
SPAWN_BOUNDARY_HYPOTENUSE = 0.3436538967 * 1.5

class PickShapenetTest(Task):

    def init_task(self) -> None:
        self.large_container = Shape('large_container')
        self.large_container.set_renderable(False)
        self.large_container.set_dynamic(False)
        self.large_container.set_respondable(False)
        self.large_container.set_collidable(False)

        self.success_detector = ProximitySensor('pick_and_lift_success')
        self.success_detector.set_position([.278466582, -.00815787166, 1.47197676])
        self.target_waypoint = Dummy('waypoint3')
        self.spawn_boundary_shape = Shape('spawn_boundary')
        self.spawn_boundary = SpawnBoundary([self.spawn_boundary_shape])
        self.register_waypoint_ability_start(1, self._move_above_object)
        self.register_waypoints_should_repeat(self._repeat)
        self.num_objects = 1
        # self.main_shapenet_sampler = ShapenetSampler(["mug",])
        # self.extra_shapenet_sampler = ShapenetSampler(["lamp", "clock"])

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index   
        """         
        from rlbench.backend.shapenet_utils import spawn_temp
        self.bin_objects = spawn_temp(self.get_base())
        # self.bin_objects = self.main_shapenet_sampler.sample_shapenet_objects(self.get_base(), 1)
        # self.bin_objects.extend(self.extra_shapenet_sampler.sample_shapenet_objects(self.get_base(), self.num_objects-1))
        self.register_graspable_objects(self.bin_objects)
        self.spawn_boundary.clear()
        self.large_container.set_position(
                [0.0, 0.0, 0.01], relative_to=self.large_container,
                reset_dynamics=True)
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
        """
        return ['pick and lift and object from the container']

    def variation_count(self) -> int:
        return len(colors)
    
    def set_num_objects(self, num_objects) -> None:
        self.num_objects = num_objects

    def change_reward(self, task) -> None:
        self.reward_lang = task
    
    # def spawn_remaining_objects(self) -> None:
    #     for ob in self.bin_objects[1:]:
    #         ob.set_position(
    #             [0.0, 0.0, 0.2], relative_to=self.large_container,
    #             reset_dynamics=True)
    #         self.spawn_boundary.sample(
    #             ob, ignore_collisions=False, min_distance=0.05)

    def any_object_is_grasped(self) -> bool:
        for cond in self.conditions: 
            if cond.condition_met()[0]:
                return True
        return False

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def reward(self) -> float:
        reward = 0
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
