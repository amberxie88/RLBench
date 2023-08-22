import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import PickShapenetTest, PickShapenetObjects
import time
from rlbench.observation_config import ObservationConfig

from PIL import Image

from pyrep import PyRep

def move(env, agent, index, delta, pos, quat):
    pos[index] += delta
    new_joint_angles = agent.solve_ik_via_jacobian(pos, quaternion=quat)
    agent.set_joint_target_positions(new_joint_angles)
    env._pyrep.step()

action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(False),
            gripper_action_mode=Discrete(),
            )

# we only support reach_target in this codebase
obs_config = ObservationConfig()

## Camera setups
obs_config.front_camera.set_all(False)
obs_config.wrist_camera.set_all(False)
obs_config.left_shoulder_camera.set_all(False)
obs_config.right_shoulder_camera.set_all(False)
obs_config.overhead_camera.set_all(False)

obs_config.joint_forces = False
obs_config.joint_positions = True
obs_config.joint_velocities = True
obs_config.task_low_dim_state = True
obs_config.gripper_touch_forces = False
obs_config.gripper_pose = True
obs_config.gripper_open = True
obs_config.gripper_matrix = False
obs_config.gripper_joint_positions = True
      
env = Environment(
            action_mode=action_mode,
            shaped_rewards=True,
            obs_config=obs_config,
        )
env.launch()

task = env.get_task(PickShapenetObjects)

# Target relative to container
target = np.array([0.3, 0., 0.2])

max_delta = np.array([0.03, 0.03, 0.03])

for _ in range(5):
  descriptions, obs = task.reset()
  starting_joint_positions = task._robot.arm.get_joint_positions()
  pos = task._robot.arm.get_tip().get_position()
  quat = task._robot.arm.get_tip().get_quaternion()
  print("starting joint positions: ", starting_joint_positions)

  container_pos = task._task.large_container.get_position()
  print("target pos: ", container_pos + target)
  for i in range(100):
    
    print("Step i - pos: ", task._robot.arm.get_tip().get_position())
    
    distance = container_pos + target - pos

    delta = np.minimum(np.abs(distance),max_delta)*np.sign(distance)

    action = np.zeros(env.action_shape) 
    action[6] = 1.

    # if abs(delta[2]) > 1e-2:
    #   delta[:2] = 0
    # elif abs(delta[0]) > 1e-2:
    #   delta[1:] = 0
    # elif abs(delta[1]) > 1e-2:
    #   delta[0] = 0
    #   delta[2] = 0
    # else:
      # action[3:7] = np.array([0,1,0,0])

    pos += delta

    print("next pos: ", pos)
    print(delta)
    # print("quat: ", action[3:7])

    action[:3] = delta

    obs, reward, terminate = task.step(action)


time.sleep(0.5)
