# this script retargets the hand from the input from the user to the joint angles for the desired hand

import os
import sys
import time
import yaml 
import torch
import argparse
import numpy as np
import genesis as gs
from os.path import join
from pathlib import Path
from copy import deepcopy

from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.demo_data import get_demo_data 
from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg 
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.constructors import get_common_argparser, parse_clip_string  
from dexmachina.retargeting.retarget_utils import compose_retarget_config, retarget_all_steps

# need `pip install dex-retargeting`
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.kinematics_adaptor import KinematicAdaptor, MimicJointKinematicAdaptor

PROCESSED_DATADIR=get_asset_path("arctic/processed")
RETARGET_DIR=get_asset_path("retargeted")
RETARGETER_RESULTS_DIR=get_asset_path("retargeter_results/pure_retarget")