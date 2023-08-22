import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import PickShapenetTest, PickShapenetObjects
import time

from PIL import Image

action_mode = MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete(),
            )
env = Environment(
            action_mode=action_mode,
            shaped_rewards=True,
        )
env.launch()

# task = env.get_task(PickShapenetObjects)
task = env.get_task(PickShapenetTest)
for _ in range(5):
  descriptions, obs = task.reset()
  for _ in range(3):
    obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))
    time.sleep(1)
  # breakpoint()
  # optional: save as model to debug
  # task._task.get_base().save_model("/desired/path")
  # optional: check image
  # obs = env._scene.get_observation()
  # image_pil = Image.fromarray(obs.wrist_rgb)
  # image_pil.save("/desired/path")

time.sleep(0.5)
