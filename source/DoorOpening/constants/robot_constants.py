import torch
import numpy as np

CAMERA_JOINT_DEFAULT_VALUES = {
    "x5_joint1": 0.0, 
    "x5_joint2": 0.15, 
    "x5_joint3": 0.2, 
    "x5_joint4": 0, 
    "x5_joint5": 0.0, 
    "x5_joint6": 0.0,
}

# CAMERA_JOINT_DEFAULT_VALUES = {
#     "x5_joint1": 0.0,
#     "x5_joint2": 0,
#     "x5_joint3": 0,
#     "x5_joint4": 0.2,
#     "x5_joint5": 0.0,
#     "x5_joint6": 1.57,
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

FRANKA_END_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.25,
    "panda_joint3": 0.0,
    "panda_joint4": -0.5 * np.pi,
    "panda_joint5": 0.0,
    "panda_joint6": 0.5 * np.pi,
    "panda_joint7": 0.0,
}

DEFAULT_JOINT_POS = {
    # "x5_joint1": 2.2, 
    # "x5_joint2": 2.355, 
    # "x5_joint3": 0.45, 
    # "x5_joint4": 1.57, 
    # "x5_joint5": 0.0, 
    # "x5_joint6": 0.0,
    "x5_joint1": 0.0, 
    "x5_joint2": 0, 
    "x5_joint3": 0, 
    "x5_joint4": 0, 
    "x5_joint5": 0.0, 
    "x5_joint6": 0.0,
    "panda_joint1": 0.0,
    "panda_joint2": -0.25 * np.pi,
    "panda_joint3": 0.0,
    "panda_joint4": -0.75 * np.pi,
    "panda_joint5": 0.0,
    "panda_joint6": 0.5 * np.pi,
    "panda_joint7": 0.0,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
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

CAMERA_JOINT_NAMES = [
    "x5_joint1", 
    "x5_joint2", 
    "x5_joint3", 
    "x5_joint4", 
    "x5_joint5", 
    "x5_joint6",
]

BASE_JOINT_NAMES = [
    'base_x_joint',
    'base_y_joint',
    'base_rotation_joint',
]

DM_JOINT_NAMES = BASE_JOINT_NAMES + FRANKA_JOINT_NAMES
DEBUG_JOINT_NAMES = BASE_JOINT_NAMES + FRANKA_JOINT_NAMES + list(CAMERA_JOINT_DEFAULT_VALUES.keys())

# The Franka hand's two prismatic DOFs. They are ONE actuator on the real gripper, so
# panda_finger_joint2 mimics panda_finger_joint1 1:1 (see the <mimic> tag in glorbot.urdf) and only
# the driven joint is ever commanded -- use DRIVEN_FINGER_JOINT_NAME / FULL_JOINT_NAMES for that.
# This list is both DOFs, for state reads and actuator/gain lookups.
DRIVEN_FINGER_JOINT_NAME = 'panda_finger_joint1'
MIMIC_FINGER_JOINT_NAME = 'panda_finger_joint2'

FINGER_JOINT_NAMES = [
    DRIVEN_FINGER_JOINT_NAME,
    MIMIC_FINGER_JOINT_NAME,
]

# The COMMANDED joints. The gripper contributes one entry, not two: the follower tracks through
# the mimic coupling and must never be given a target of its own. Use ALL_DOF_NAMES when you need
# every articulation DOF (e.g. sizing a full joint-state tensor).
FULL_JOINT_NAMES = (
    BASE_JOINT_NAMES + FRANKA_JOINT_NAMES + [DRIVEN_FINGER_JOINT_NAME] + CAMERA_JOINT_NAMES
)

ALL_DOF_NAMES = BASE_JOINT_NAMES + FRANKA_JOINT_NAMES + FINGER_JOINT_NAMES + CAMERA_JOINT_NAMES

ROBOT_KEY_BODY_NAMES = [
    "tidybot2_base_link",
    "panda_link2",
    "panda_link4",
    "panda_link6",
    "panda_hand",
    # "x5_camera_link",
]

ROBOT_RESET_KEY_BODY_NAMES = [
    "tidybot2_base_link",
    "panda_link4",
    "panda_hand",
    # "x5_camera_link",
]

ROBOT_PALM_LINK_NAME = "panda_hand"
ROBOT_BASE_BODY_LINK_NAME = "tidybot2_base_link"
# Pad frames on the two fingers, i.e. the grasp keypoints.
ROBOT_FINGERTIP_BODY_NAMES = ["left_fingertip", "right_fingertip"]

# Gripper half-opening in metres: 0.04 is fully open (8 cm between the pads), 0.0 is closed.
# Both DOFs are written so the fingers stay symmetric whether or not the mimic constraint exists.
GRIPPER_OPEN_WIDTH = 0.04
GRIPPER_CLOSED_WIDTH = 0.0

CLOSE_FINGER_JOINT_VALUES = {name: GRIPPER_CLOSED_WIDTH for name in FINGER_JOINT_NAMES}

OPEN_FINGER_JOINT_VALUES = {name: GRIPPER_OPEN_WIDTH for name in FINGER_JOINT_NAMES}
