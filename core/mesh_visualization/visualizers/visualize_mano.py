""" Visualizes right hand from MANO on random data """

import argparse
import mano
import torch

from core.mesh_visualization.mesh_viewer import Mesh
from tools.cfg_parser import Config


def visualize_mano(cfg):
    n_comps = 45
    batch_size = 10

    rh_model = mano.load(model_path=cfg.model_path,
                         is_right=True,
                         num_pca_comps=n_comps,
                         batch_size=batch_size,
                         flat_hand_mean=False)

    betas = torch.rand(batch_size, 10) * .1
    pose = torch.rand(batch_size, n_comps) * .1
    global_orient = torch.rand(batch_size, 3)
    transl = torch.rand(batch_size, 3)

    output = rh_model(betas=betas,
                      global_orient=global_orient,
                      hand_pose=pose,
                      transl=transl,
                      return_verts=True,
                      return_tips=True)

    h_meshes = rh_model.hand_meshes(output)
    j_meshes = rh_model.joint_meshes(output)

    # visualize hand mesh only
    h_meshes[0].show()

    # visualize joints mesh only
    j_meshes[0].show()

    # visualize hand and joint meshes
    hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])
    hj_meshes.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MANO-visualize')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing mano models')

    args = parser.parse_args()
    model_path = args.model_path

    cfg = {
        'model_path': model_path,
    }

    cfg = Config(**cfg)
    visualize_mano(cfg)
