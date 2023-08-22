from typing import List
import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType, TextureMappingMode
from pyrep.objects.shape import Shape

from rlbench.backend.scene import Scene
from rlbench.observation_config import ObservationConfig
from rlbench.backend.robot import Robot
from rlbench.sim2real.domain_randomization import RandomizeEvery

SCENE_OBJECTS = ['Floor', 'Roof', 'Wall1', 'Wall2', 'Wall3', 'Wall4',
                 'diningTable_visible']

TEX_KWARGS = {
    'mapping_mode': TextureMappingMode.PLANE,
    'repeat_along_u': False,
    'repeat_along_v': False,
    'uv_scaling': [10., 10.]
}


class ObjectwiseRandomizationScene(Scene):

    '''
    visual_randomization_config should be a list of three elements for [table, walls, floor]
    '''

    def __init__(self,
                 pyrep: PyRep,
                 robot: Robot,
                 obs_config: ObservationConfig = ObservationConfig(),
                 robot_setup: str = 'Panda',
                 randomize_every: RandomizeEvery=RandomizeEvery.EPISODE,
                 frequency: int=1,
                 visual_randomization_config=None,
                 dynamics_randomization_config=None,
                 tex_kwargs=None):
        super().__init__(pyrep, robot, obs_config, robot_setup)
        self._randomize_every = randomize_every
        self._frequency = frequency
        self._visual_rand_config = visual_randomization_config
        self._dynamics_rand_config = dynamics_randomization_config
        self._tex_kwargs = tex_kwargs
        self._previous_index = -1
        self._count = 0
        self._percent_original_env = 0.2

        if self._dynamics_rand_config is not None:
            raise NotImplementedError(
                'Dynamics randomization coming soon! '
                'Only visual randomization available.')

        self._scene_objects = [Shape(name) for name in SCENE_OBJECTS]
        self._scene_objects += self.robot.arm.get_visuals()
        self._scene_objects += self.robot.gripper.get_visuals()
        self.dr_object_tracker = dict()
        if self._visual_rand_config is not None:
            # Make the floor plane renderable (to cover old floor)
            floor_copy = self._scene_objects[0].copy()
            floor_copy.set_position([0, 0, 0.01])
            floor_copy.set_renderable(False)
            self.dr_object_tracker[self._scene_objects[0].get_name()] = floor_copy

    def _should_randomize_episode(self, index: int):
        rand = self._count % self._frequency == 0 or self._count == 0
        if self._randomize_every == RandomizeEvery.VARIATION:
            if self._previous_index != index:
                self._previous_index = index
                self._count += 1
        elif self._randomize_every == RandomizeEvery.EPISODE:
            self._count += 1
        return rand

    def _randomize(self):
        tree = self.task.get_base().get_objects_in_tree(
            ObjectType.SHAPE)
        tree = [Shape(obj.get_handle()) for obj in tree + self._scene_objects]
        set_original_env = np.random.rand() < self._percent_original_env
        if self._visual_rand_config is not None:
            for config, tex_kwargs in zip(self._visual_rand_config, self._tex_kwargs):
                files = config.sample(len(tree))
                for file, obj in zip(files, tree):
                    # print('Applying texture %s to %s?' % (file, obj.get_name()))
                    if config.should_randomize(obj.get_name()):
                        if set_original_env:
                            print('Yes, resetting %s to normal' % (obj.get_name()))
                        else:
                            print('Yes, applying texture %s to %s' % (file, obj.get_name()))
                        text_ob, texture = self.pyrep.create_texture(file)
                        if "diningTable" in obj.get_name():
                            if set_original_env:
                                obj.set_renderable(True)
                                obj.set_respondable(True)
                                if obj.get_name() in self.dr_object_tracker.keys():
                                    new_obj = self.dr_object_tracker[obj.get_name()]
                                    new_obj.remove()
                                    self.dr_object_tracker.pop(obj.get_name())
                            else:
                                if obj.get_name() not in self.dr_object_tracker.keys():
                                    ungrouped = obj.ungroup()
                                    new_ungrouped = []
                                    for o in ungrouped:
                                        o.set_renderable(False)
                                        o.set_respondable(False)
                                        new_obj = o.copy()
                                        new_obj.remove_texture()
                                        new_obj.set_texture(texture, **tex_kwargs)
                                        new_obj.set_renderable(True)
                                        new_obj.set_respondable(True)
                                        new_ungrouped.append(new_obj)
                                    original_shape = self.pyrep.group_objects(ungrouped) 
                                    new_shape = self.pyrep.group_objects(new_ungrouped)
                                    self.dr_object_tracker[original_shape.get_name()] = new_shape
                                    obj.set_renderable(False)
                                    obj.set_respondable(False)
                                    new_shape.set_renderable(True)
                                else:
                                    new_shape = self.dr_object_tracker[obj.get_name()]
                                    new_ungrouped = new_shape.ungroup()
                                    for o in new_ungrouped:
                                        o.set_texture(texture, **tex_kwargs)
                                    new_shape = self.pyrep.group_objects(new_ungrouped)
                        else:
                            if set_original_env:
                                obj.set_renderable(True)
                                obj.set_respondable(True)
                                if obj.get_name() in self.dr_object_tracker.keys():
                                    new_obj = self.dr_object_tracker[obj.get_name()]
                                    new_obj.set_renderable(False)
                            else:
                                if obj.get_name() not in self.dr_object_tracker.keys():
                                    new_obj = obj.copy()
                                    obj.set_renderable(False)
                                    obj.set_respondable(False)
                                    new_obj.set_renderable(True)
                                    new_obj.set_respondable(True)
                                    new_obj.set_color([1.0,1.0,1.0])
                                    self.dr_object_tracker[obj.get_name()] = new_obj
                                else:
                                    new_obj = self.dr_object_tracker[obj.get_name()]
                                    new_obj.set_renderable(True)
                                new_obj.set_texture(texture, **tex_kwargs)
                        text_ob.remove()

    def init_task(self) -> None:
        super().init_task()

    def init_episode(self, index: int, *args, **kwargs) -> List[str]:
        ret = super().init_episode(index, *args, **kwargs)
        if (self._randomize_every != RandomizeEvery.TRANSITION and
                self._should_randomize_episode(index)):
            self._randomize()
            self.pyrep.step()  # Need to step to apply textures
        return ret

    def step(self):
        if self._randomize_every == RandomizeEvery.TRANSITION:
            if self._count % self._frequency == 0 or self._count == 0 :
                self._randomize()
            self._count += 1
        super().step()

    def reset(self) -> None:
        return super().reset()
