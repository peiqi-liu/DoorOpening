import torch
import numpy as np
CAMERA_JOINT_DEFAULT_VALUES = {
    "x5_joint1": 0.0, 
    "x5_joint2": 0.785, 
    "x5_joint3": 0.785, 
    "x5_joint4": 0.0, 
    "x5_joint5": 0.0, 
    "x5_joint6": 0.0,
}

# CAMERA_JOINT_DEFAULT_VALUES = {
#     "x5_joint1": 1.57, 
#     "x5_joint2": 2.355, 
#     "x5_joint3": 0.0, 
#     "x5_joint4": 1.57, 
#     "x5_joint5": 0.0, 
#     "x5_joint6": 0.0,
# }

FRANKA_DEFAULT_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.25 * np.pi,
    "panda_joint3": 0.0,
    "panda_joint4": -0.75 * np.pi,
    "panda_joint5": 0.0,
    "panda_joint6": 0.5 * np.pi,
    "panda_joint7": 0.0,
}

DEFAULT_JOINT_POS = {
    "x5_joint1": 2.2, 
    "x5_joint2": 2.355, 
    "x5_joint3": 0.45, 
    "x5_joint4": 1.57, 
    "x5_joint5": 0.0, 
    "x5_joint6": 0.0,
    "panda_joint1": 0.0,
    "panda_joint2": -0.25 * np.pi,
    "panda_joint3": 0.0,
    "panda_joint4": -0.75 * np.pi,
    "panda_joint5": 0.0,
    "panda_joint6": 0.5 * np.pi,
    "panda_joint7": 0.0,
    "finger_joint_12": 0.5 * np.pi,
    "finger_joint_13": 0.0,
}

FRANKA_JOINT_NAMES = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
]

BASE_JOINT_NAMES = [
    'base_x_joint',
    'base_y_joint',
    'base_rotation_joint',
]

DM_JOINT_NAMES = BASE_JOINT_NAMES + FRANKA_JOINT_NAMES
DEBUG_JOINT_NAMES = BASE_JOINT_NAMES + FRANKA_JOINT_NAMES + list(CAMERA_JOINT_DEFAULT_VALUES.keys())

FINGER_JOINT_NAMES = [
    'finger_joint_0',
    'finger_joint_1',
    'finger_joint_2',
    'finger_joint_3',
    'finger_joint_4',
    'finger_joint_5',
    'finger_joint_6',
    'finger_joint_7',
    'finger_joint_8',
    'finger_joint_9',
    'finger_joint_10',
    'finger_joint_11',
    'finger_joint_12',
    'finger_joint_13',
    'finger_joint_14',
    'finger_joint_15',
]

FULL_JOINT_NAMES = BASE_JOINT_NAMES + FRANKA_JOINT_NAMES + FINGER_JOINT_NAMES

ROBOT_KEY_BODY_NAMES = [
    "tidybot2_base_link",
    "panda_link2",
    "panda_link4",
    "panda_link6",
    "palm_lower",
]

ROBOT_RESET_KEY_BODY_NAMES = [
    "tidybot2_base_link",
    "panda_link2",
    "panda_link4",
    "panda_link6",
    "palm_lower",
]

ROBOT_PALM_LINK_NAME = "palm_lower"
ROBOT_BASE_BODY_LINK_NAME = "tidybot2_base_link"

CLOSE_FINGER_JOINT_VALUES = {
    "finger_joint_0": 0.0,
    "finger_joint_1": torch.pi / 2,
    "finger_joint_2": 1.8,
    "finger_joint_3": 1.0,
    "finger_joint_4": 0.0,
    "finger_joint_5": torch.pi / 2,
    "finger_joint_6": 1.8,
    "finger_joint_7": 1.0,
    "finger_joint_8": 0.0,
    "finger_joint_9": torch.pi / 2,
    "finger_joint_10": 1.8,
    "finger_joint_11": 1.0,
    "finger_joint_12": torch.pi / 2,
    "finger_joint_13": 0.0,
    "finger_joint_14": 0.5,
    "finger_joint_15": 1.0,
}

OPEN_FINGER_JOINT_VALUES = {
    "finger_joint_0": 0.0,
    "finger_joint_1": 0.0,
    "finger_joint_2": 0.0,
    "finger_joint_3": 0.0,
    "finger_joint_4": 0.0,
    "finger_joint_5": 0.0,
    "finger_joint_6": 0.0,
    "finger_joint_7": 0.0,
    "finger_joint_8": 0.0,
    "finger_joint_9": 0.0,
    "finger_joint_10": 0.0,
    "finger_joint_11": 0.0,
    "finger_joint_12": torch.pi / 2,
    "finger_joint_13": 0.0,
    "finger_joint_14": 0.0,
    "finger_joint_15": 0.0,
}
