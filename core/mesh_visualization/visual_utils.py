""" Utility functions for visualizing data """

import numpy as np


def euler(rots, order='xyz', units='deg'):
    rots = np.asarray(rots)
    single_val = False if len(rots.shape) > 1 else True
    rots = rots.reshape(-1, 3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz, order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis == 'x':
                r = np.dot(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), r)
            if axis == 'y':
                r = np.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), r)
            if axis == 'z':
                r = np.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats


# colors used for visualizations
colors = {
    'pink': [1.00, 0.75, 0.80],
    'skin': [0.96, 0.75, 0.69],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [.0, .0, 1.],
    'white': [1., 1., 1.],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0., 0., 0.],
}

# mapping the contact ids to each body part in SMPLX
contact_ids = {'Body': 1,
               'L_Thigh': 2,
               'R_Thigh': 3,
               'Spine': 4,
               'L_Calf': 5,
               'R_Calf': 6,
               'Spine1': 7,
               'L_Foot': 8,
               'R_Foot': 9,
               'Spine2': 10,
               'L_Toes': 11,
               'R_Toes': 12,
               'Neck': 13,
               'L_Shoulder': 14,
               'R_Shoulder': 15,
               'Head': 16,
               'L_UpperArm': 17,
               'R_UpperArm': 18,
               'L_ForeArm': 19,
               'R_ForeArm': 20,
               'L_Hand': 21,
               'R_Hand': 22,
               'Jaw': 23,
               'L_Eye': 24,
               'R_Eye': 25,
               'L_Index1': 26,
               'L_Index2': 27,
               'L_Index3': 28,
               'L_Middle1': 29,
               'L_Middle2': 30,
               'L_Middle3': 31,
               'L_Pinky1': 32,
               'L_Pinky2': 33,
               'L_Pinky3': 34,
               'L_Ring1': 35,
               'L_Ring2': 36,
               'L_Ring3': 37,
               'L_Thumb1': 38,
               'L_Thumb2': 39,
               'L_Thumb3': 40,
               'R_Index1': 41,
               'R_Index2': 42,
               'R_Index3': 43,
               'R_Middle1': 44,
               'R_Middle2': 45,
               'R_Middle3': 46,
               'R_Pinky1': 47,
               'R_Pinky2': 48,
               'R_Pinky3': 49,
               'R_Ring1': 50,
               'R_Ring2': 51,
               'R_Ring3': 52,
               'R_Thumb1': 53,
               'R_Thumb2': 54,
               'R_Thumb3': 55}

# mapping the contact ids to the right hand in MANO
rhand_contact_ids = {'R_Hand': 22,
                     'R_Index1': 41,
                     'R_Index2': 42,
                     'R_Index3': 43,
                     'R_Middle1': 44,
                     'R_Middle2': 45,
                     'R_Middle3': 46,
                     'R_Pinky1': 47,
                     'R_Pinky2': 48,
                     'R_Pinky3': 49,
                     'R_Ring1': 50,
                     'R_Ring2': 51,
                     'R_Ring3': 52,
                     'R_Thumb1': 53,
                     'R_Thumb2': 54,
                     'R_Thumb3': 55}

# mapping the contact ids to the left hand in MANO
lhand_contact_ids = {'L_Hand': 21,
                     'L_Index1': 26,
                     'L_Index2': 27,
                     'L_Index3': 28,
                     'L_Middle1': 29,
                     'L_Middle2': 30,
                     'L_Middle3': 31,
                     'L_Pinky1': 32,
                     'L_Pinky2': 33,
                     'L_Pinky3': 34,
                     'L_Ring1': 35,
                     'L_Ring2': 36,
                     'L_Ring3': 37,
                     'L_Thumb1': 38,
                     'L_Thumb2': 39,
                     'L_Thumb3': 40}
