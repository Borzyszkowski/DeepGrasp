""" Renders whole-body from GRAB as .gif, .mp4 or .png snapshots """

import argparse
import glob
import numpy as np
import os
import smplx
import torch

from tqdm import tqdm

from core.mesh_visualization.mesh_viewer import Mesh, MeshViewer
from core.mesh_visualization.object_model import ObjectModel
from core.mesh_visualization.visual_utils import colors, euler
from tools.cfg_parser import Config
from tools.utils import (makepath, params2torch, parse_npz, to_cpu)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def render_whole_body_interaction(cfg):
    """ Create visualization for rendering of the whole body hand in MeshViewer. """
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

    sbj_mesh = os.path.join(grab_path, '..', seq_data.body.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

    sbj_m = smplx.create(model_path=cfg.model_path,
                         model_type='smplx',
                         gender=gender,
                         num_pca_comps=n_comps,
                         v_template=sbj_vtemp,
                         batch_size=T)

    sbj_parms = params2torch(seq_data.body.params)
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)
    joints_sbj = to_cpu(sbj_m(**sbj_parms).joints)

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

    seq_render_path = makepath(sequence.replace('.npz', '').replace(cfg.grab_path, cfg.render_path))

    skip_frame = 4
    print(f"Rendering data for {seq_data['obj_name']}_{seq_data['motion_intent']}")
    for frame in tqdm(range(0, T, skip_frame)):
        o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
        o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)
        s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
        s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)
        s_mesh_wf = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['grey'], wireframe=True)
        t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])
        j_mesh = Mesh(vertices=joints_sbj[frame], vc=colors['green'], smooth=True)
        mv.set_static_meshes([o_mesh, j_mesh, s_mesh, s_mesh_wf, t_mesh])
        if render_format == 'mp4':
            mv.save_snapshot(seq_render_path + '/%04d.png' % frame)

    if render_format == 'mp4':
        mv.save_recording(seq_render_path)
    elif render_format == 'gif':
        mv.close_viewer()
        mv.end_gif_recording(seq_render_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRAB-render')
    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--render-path', required=True, type=str,
                        help='The path to the folder to save the renderings')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')
    parser.add_argument('--render-format', required=False, type=str, choices=['gif', 'mp4'], default='gif',
                        help='The path to the folder containing smplx models')

    args = parser.parse_args()

    grab_path = args.grab_path
    render_path = args.render_path
    model_path = args.model_path
    render_format = args.render_format

    cfg = {
        'grab_path': grab_path,
        'model_path': model_path,
        'render_path': render_path,
        'render_format': render_format
    }

    cfg = Config(**cfg)
    render_whole_body_interaction(cfg)
