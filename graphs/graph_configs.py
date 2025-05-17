""" Collection of graph configurations for different body models. """

# SMPL joint names, in the original order (from pelvis to head)
SMPL_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand']

# MANO joint names, in the original order (from palm to fingertip)
MANO_JOINT_NAMES = [
    'wrist',
    'index1',
    'index2',
    'index3',
    'middle1',
    'middle2',
    'middle3',
    'pinky1',
    'pinky2',
    'pinky3',
    'ring1',
    'ring2',
    'ring3',
    'thumb1',
    'thumb2',
    'thumb3',
    'thumb_tip',
    'index_tip',
    'middle_tip',
    'ring_tip',
    'pinky_tip',
]

# MANO idexes of the finger tips in the mesh
TIP_IDS = {
    'mano': {
        'thumb': 744,
        'index': 320,
        'middle': 443,
        'ring': 554,
        'pinky': 671,
    }
}

# Definition of the graph edges for the MANO hand
JOINT_GRAPH = {
    'wrist': 'index1',
    'index1': 'index2',
    'index2': 'index3',
    'wrist': 'middle1',
    'middle1': 'middle2',
    'middle2': 'middle3',
    'wrist': 'pinky1',
    'pinky1': 'pinky2',
    'pinky2': 'pinky3',
    'wrist': 'ring1',
    'ring1': 'ring2',
    'ring2': 'ring3',
    'wrist': 'thumb1',
    'thumb1': 'thumb2',
    'thumb2': 'thumb3',
    'thumb3': 'thumb_tip',
    'index3': 'index_tip',
    'middle3': 'middle_tip',
    'pinky3': 'pinky_tip',
    'ring3': 'ring_tip',
}
