""" Renders both hands from GRAB as .gif, .mp4 or .png snapshots """

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
from core.mesh_visualization.visual_utils import colors, euler, lhand_contact_ids, rhand_contact_ids
from tools.cfg_parser import Config
from tools.utils import (makepath, params2torch, parse_npz, to_cpu)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def render_rlhands_interaction(cfg):
    """ Create visualization for rendering of the right and left hand in MeshViewer. """
    grab_path = cfg.grab_path
    all_seqs = glob.glob(grab_path + '/*/*.npz')

    choice = np.random.choice(len(all_seqs), 10, replace=False)
    for i in tqdm(choice):
        offscreen = False if render_format == 'gif' else True
        mv = MeshViewer(width=1600, height=1200, offscreen=offscreen)

        # set the camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
        camera_pose[:3, 3] = np.array([-.5, -1.4, 1.5])
        mv.update_camera_pose(camera_pose)

        vis_sequence(cfg, all_seqs[i], mv)


def vis_sequence(cfg, sequence, mv):
    """ Visualize given sequence of events, according to the configuration. """
    if render_format == 'gif':
        mv.start_gif_recording()

    seq_data = parse_npz(sequence)
    n_comps = seq_data['n_comps']
    gender = seq_data['gender']
    T = seq_data.n_frames

    sbj_rh_mesh = os.path.join(grab_path, '..', seq_data.rhand.vtemp)
    sbj_rh_vtemp = np.array(Mesh(filename=sbj_rh_mesh).vertices)
    sbj_lh_mesh = os.path.join(grab_path, '..', seq_data.lhand.vtemp)
    sbj_lh_vtemp = np.array(Mesh(filename=sbj_lh_mesh).vertices)

    sbj_rh_m = smplx.create(model_path=cfg.model_path_right,
                            model_type='mano',
                            gender=gender,
                            num_pca_comps=n_comps,
                            v_template=sbj_rh_vtemp,
                            batch_size=T,
                            flat_hand_mean=True)
    sbj_rh_parms = params2torch(seq_data.rhand.params)
    verts_rh_sbj = to_cpu(sbj_rh_m(**sbj_rh_parms).vertices)

    sbj_lh_m = smplx.create(model_path=cfg.model_path_left,
                            model_type='mano',
                            is_rhand=False,
                            gender=gender,
                            num_pca_comps=n_comps,
                            v_template=sbj_lh_vtemp,
                            batch_size=T,
                            flat_hand_mean=True)
    sbj_lh_parms = params2torch(seq_data.lhand.params)
    verts_lh_sbj = to_cpu(sbj_lh_m(**sbj_lh_parms).vertices)

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
    lhand_vertices = vertex_ids_mapping['left_hand']

    seq_render_path = makepath(sequence.replace('.npz', '').replace(cfg.grab_path, cfg.render_path))

    skip_frame = 4
    print(f"Rendering data for {seq_data['obj_name']}_{seq_data['motion_intent']}")
    for frame in tqdm(range(0, T, skip_frame)):
        o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
        cont_rhand = np.vectorize(lambda x: x in rhand_contact_ids.values())(seq_data['contact']['object'][frame])
        cont_lhand = np.vectorize(lambda x: x in lhand_contact_ids.values())(seq_data['contact']['object'][frame])
        obj_sbj_contacts = cont_rhand + cont_lhand
        o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=obj_sbj_contacts)

        s_rh_mesh = Mesh(vertices=verts_rh_sbj[frame], faces=sbj_rh_m.faces, vc=colors['pink'], smooth=True)
        s_rh_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame][rhand_vertices] > 0)
        s_rh_mesh_wf = Mesh(vertices=verts_rh_sbj[frame], faces=sbj_rh_m.faces, vc=colors['grey'], wireframe=True)

        s_lh_mesh = Mesh(vertices=verts_lh_sbj[frame], faces=sbj_lh_m.faces, vc=colors['pink'], smooth=True)
        s_lh_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame][lhand_vertices] > 0)
        s_lh_mesh_wf = Mesh(vertices=verts_lh_sbj[frame], faces=sbj_lh_m.faces, vc=colors['grey'], wireframe=True)

        t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

        mv.set_static_meshes([o_mesh, s_rh_mesh, s_rh_mesh_wf, s_lh_mesh, s_lh_mesh_wf, t_mesh])
        if render_format == 'mp4':
            mv.save_snapshot(seq_render_path + '/%04d.png' % frame)

    if render_format == 'mp4':
        mv.save_recording(seq_render_path)
    elif render_format == 'gif':
        mv.close_viewer()
        mv.end_gif_recording(seq_render_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Both-Hands-Interaction-render')
    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--render-path', required=True, type=str,
                        help='The path to the folder to save the renderings')
    parser.add_argument('--model-path-right', required=True, type=str,
                        help='The path to the folder containing MANO right hand model')
    parser.add_argument('--model-path-left', required=True, type=str,
                        help='The path to the folder containing MANO left hand model')
    parser.add_argument('--render-format', required=False, type=str, choices=['gif', 'mp4'], default='gif',
                        help='Format of renderings specified as gif or mp4')

    args = parser.parse_args()
    grab_path = args.grab_path
    render_path = args.render_path
    model_path_right = args.model_path_right
    model_path_left = args.model_path_left
    render_format = args.render_format

    cfg = {
        'grab_path': grab_path,
        'model_path_right': model_path_right,
        'model_path_left': model_path_left,
        'render_path': render_path,
        'render_format': render_format
    }

    cfg = Config(**cfg)
    render_rlhands_interaction(cfg)
