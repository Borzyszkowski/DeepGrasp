""" Visualizes right hand from GRAB in MeshViewer """

import argparse
import glob
import numpy as np
import os
import pickle
import smplx
import torch

from tqdm import tqdm

from core.mesh_visualization.mesh_viewer import Mesh, MeshViewer
from core.mesh_visualization.object_model import ObjectModel
from core.mesh_visualization.visual_utils import colors, euler, rhand_contact_ids
from tools.cfg_parser import Config
from tools.utils import params2torch, parse_npz, to_cpu, to_np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def joints_to_mesh(output, frame, radius=.002, vc=colors['green']):
    """ Calculate joints for each frame.
    By default, there are 15 hand joints and 1 body (wrist) joint. It is possible to compute 5 extra joints in MANO:
    https://github.com/vchoutas/smplx/blob/master/smplx/body_models.py#L1680 """
    joints = to_np(output.joints)

    if joints.ndim < 3:
        joints = joints.reshape(1, -1, 3)

    joint_mesh = Mesh(vertices=joints[frame], radius=radius, vc=vc)
    return joint_mesh


def visualize_rhand_interaction(cfg):
    """ Create visualization of the right hand in MeshViewer. """
    grab_path = cfg.grab_path
    all_seqs = glob.glob(grab_path + '/*/*.npz')

    mv = MeshViewer(offscreen=False)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
    mv.update_camera_pose(camera_pose)

    choice = np.random.choice(len(all_seqs), 10, replace=False)
    for i in tqdm(choice):
        vis_sequence(cfg, all_seqs[i], mv)
    mv.close_viewer()


def vis_sequence(cfg, sequence, mv):
    """ Visualize given sequence of events, according to the configuration. """
    seq_data = parse_npz(sequence)
    n_comps = seq_data['n_comps']
    gender = seq_data['gender']
    T = seq_data.n_frames

    sbj_mesh = os.path.join(grab_path, '..', seq_data.rhand.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

    sbj_m = smplx.create(model_path=cfg.model_path,
                         model_type='mano',
                         gender=gender,
                         num_pca_comps=n_comps,
                         v_template=sbj_vtemp,
                         batch_size=T,
                         flat_hand_mean=True)
    sbj_parms = params2torch(seq_data.rhand.params)
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)

    if cfg.compute_joints:
        # Represent values in Torch to get the joint mesh
        torch_global_orient = torch.from_numpy(seq_data.rhand.params.global_orient)
        torch_transl = torch.from_numpy(seq_data.rhand.params.transl)
        torch_hand_pose = torch.from_numpy(seq_data.rhand.params.hand_pose)

        output = sbj_m(global_orient=torch_global_orient,
                       hand_pose=torch_hand_pose,
                       transl=torch_transl,
                       return_verts=True,
                       return_tips=True)

    obj_mesh = os.path.join(grab_path, '..', seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh)

    obj_vtemp = np.array(obj_mesh.vertices)
    obj_m = ObjectModel(v_template=obj_vtemp,
                        batch_size=T)
    obj_parms = params2torch(seq_data.object.params)
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)

    table_mesh = os.path.join(grab_path, '..', seq_data.table.table_mesh)
    table_mesh = Mesh(filename=table_mesh)
    table_vtemp = np.array(table_mesh.vertices)
    table_m = ObjectModel(v_template=table_vtemp,
                          batch_size=T)
    table_parms = params2torch(seq_data.table.params)
    verts_table = to_cpu(table_m(**table_parms).vertices)

    vertex_ids_mapping = pickle.load(open('core/mesh_visualization/MANO_SMPLX_vertex_ids.pkl', 'rb'))
    rhand_vertices = vertex_ids_mapping['right_hand']

    skip_frame = 4
    for frame in range(0, T, skip_frame):
        o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
        obj_sbj_contacts = np.vectorize(lambda x: x in rhand_contact_ids.values())(seq_data['contact']['object'][frame])
        o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=obj_sbj_contacts)

        s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
        s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame][rhand_vertices] > 0)
        s_mesh_wf = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['grey'], wireframe=True)

        t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

        if cfg.compute_joints:
            j_mesh = joints_to_mesh(output, frame)
            mv.set_static_meshes([o_mesh, s_mesh, s_mesh_wf, j_mesh, t_mesh])
        else:
            mv.set_static_meshes([o_mesh, s_mesh, s_mesh_wf, t_mesh])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Right-Hand-Interaction-visualize')
    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing MANO right hand model')
    parser.add_argument('--compute-joints', required=None, type=bool, default=True,
                        help='Determines if the joins should be computed and visualised as a mesh.')

    args = parser.parse_args()
    grab_path = args.grab_path
    model_path = args.model_path
    compute_joints = args.compute_joints

    cfg = {
        'grab_path': grab_path,
        'model_path': model_path,
        'compute_joints': compute_joints,
    }

    cfg = Config(**cfg)
    visualize_rhand_interaction(cfg)
