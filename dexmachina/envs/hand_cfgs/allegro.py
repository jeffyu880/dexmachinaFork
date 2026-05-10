import os  
from os.path import join
from dexmachina.asset_utils import get_urdf_path

allegro_asset_dir = "allegro_hand/"
left_rel_urdf = join(allegro_asset_dir, "allegro_hand_left_6dof.urdf") 
right_rel_urdf = join(allegro_asset_dir, "allegro_hand_right_6dof.urdf")

# left kpt_names (raw, 25 entries):
#   0: link_15.0_tip
#   1: link_11.0_tip
#   2: link_7.0_tip
#   3: link_3.0_tip
#   4: base_link
#   5: link_0.0
#   6: link_4.0
#   7: link_8.0
#   8: link_12.0
#   9: link_1.0
#   10: link_5.0
#   11: link_9.0
#   12: link_13.0
#   13: link_2.0
#   14: link_6.0
#   15: link_10.0
#   16: link_14.0
#   17: link_3.0
#   18: link_7.0
#   19: link_11.0
#   20: link_15.0

# using 1 indexing to store indices for the kpt values

ALLEGRO_LEFT_CFG={
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "base_dummy_link",
    "kpt_link_names": ["link_15.0_tip", "link_11.0_tip", "link_7.0_tip", "link_3.0_tip" ],
    
    "keypoint_idx" : {
            "thumb_tip": [1],              # link_15.0_tip
            "index_tip": [2],              # link_11.0_tip
            "middle_tip": [3],             # link_7.0_tip
            "pinky_tip": [4],              # link_3.0_tip
            "level_1_joints": [6,  7,  8,  9,  10, 11, 12, 13],    # indices 5-8: link_0.0, link_4.0, link_8.0, link_12.0
            "level_2_joints": [14, 15, 16, 17, 18, 19, 20, 21], # indices 9-12: link_1.0, link_5.0, link_9.0, link_13.0
        },
    "actuators": {
        "finger": dict(
            joint_exprs=['.*.0.+'],
            kp=30.0,
            kv=2.0,
            force_range=100.0,
        ),
        "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=60,
            kv=5.0,
            force_range=100.0,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=350.0,
            kv=20.0,
            force_range=100.0,
        ),
    },
    "collision_groups": {8: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 1, 18: 2, 19: 3, 20: 4, 21: 1, 22: 2, 23: 3, 24: 4, 25: 1, 26: 2, 27: 3, 28: 4},
    "collision_palm_name": "base_link",
}

# right kpt_names (raw, 25 entries):
#   0: link_15.0_tip
#   1: link_3.0_tip
#   2: link_7.0_tip
#   3: link_11.0_tip
#   4: base_link
#   5: link_0.0
#   6: link_4.0
#   7: link_8.0
#   8: link_12.0
#   9: link_1.0
#   10: link_5.0
#   11: link_9.0
#   12: link_13.0
#   13: link_2.0
#   14: link_6.0
#   15: link_10.0
#   16: link_14.0
#   17: link_3.0
#   18: link_7.0
#   19: link_11.0
#   20: link_15.0

ALLEGRO_RIGHT_CFG={
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "base_dummy_link",
    "kpt_link_names": ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip" ],
    
    "keypoint_idx" : {
            "thumb_tip": [1],              # link_15.0_tip
            "index_tip": [2],              # link_3.0_tip
            "middle_tip": [3],             # link_7.0_tip
            "pinky_tip": [4],              # link_11.0_tip
            "level_1_joints": [6,  7,  8,  9,  10, 11, 12, 13],    # indices 5-8: link_0.0, link_4.0, link_8.0, link_12.0
            "level_2_joints": [14, 15, 16, 17, 18, 19, 20, 21], # indices 9-12: link_1.0, link_5.0, link_9.0, link_13.0
    }, 
    "actuators": ALLEGRO_LEFT_CFG["actuators"].copy(),
    "collision_groups": {8: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 1, 18: 2, 19: 3, 20: 4, 21: 1, 22: 2, 23: 3, 24: 4, 25: 1, 26: 2, 27: 3, 28: 4},
    "collision_palm_name": "base_link",
}

ALLEGRO_CFGs=dict(
    left=ALLEGRO_LEFT_CFG,
    right=ALLEGRO_RIGHT_CFG,
)

# class Allegro:
#     """Allegro hand configuration and utilities."""
    
#     def __init__(self):
#         """Initialize Allegro hand parameters."""
#         # Index mapping for keypoints in kpt_pos order
#         # Order of storing kpts in 
#         # ['link_15.0_tip', 'link_3.0_tip', 'link_7.0_tip', 
#         #   'link_11.0_tip', 'base_link',    'link_0.0', 
#         #   'link_4.0',       'link_8.0', 'link_12.0', 
#         #    'link_1.0',     'link_5.0',  'link_9.0', 
#         #     'link_13.0', 'link_2.0', 'link_6.0', 
#         #     'link_10.0', 'link_14.0', 'link_3.0', 
#         #    'link_7.0', 'link_11.0', 'link_15.0']
        
#         # using 1 indexing to store indices for the kpt values
#         self.right_
 
        
#         self.body_names = [
#             "base_link",
#             # pinky
#             "link_0.0",
#             "link_1.0",
#             "link_2.0",
#             "link_3.0",
#             "link_3.0_tip",
#             # thumb
#             "link_12.0",
#             "link_13.0",
#             "link_14.0",
#             "link_15.0",
#             "link_15.0_tip",
#             # middle
#             "link_4.0",
#             "link_5.0",
#             "link_6.0",
#             "link_7.0",
#             "link_7.0_tip",
#             # index
#             "link_8.0",
#             "link_9.0",
#             "link_10.0",
#             "link_11.0",
#             "link_11.0_tip",
#         ]
#         self.dof_names = [
#             "joint_0.0",
#             "joint_1.0",
#             "joint_2.0",
#             "joint_3.0",
#             "joint_12.0",
#             "joint_13.0",
#             "joint_14.0",
#             "joint_15.0",
#             "joint_4.0",
#             "joint_5.0",
#             "joint_6.0",
#             "joint_7.0",
#             "joint_8.0",
#             "joint_9.0",
#             "joint_10.0",
#             "joint_11.0",
#         ]