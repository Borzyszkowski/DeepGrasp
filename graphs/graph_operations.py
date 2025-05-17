""" Collection of functions for graph operations """

import torch

from tools.utils import get_project_root


def compute_vertex_graph(triangular_faces):
    """ Computes a graph for 778 vertices of the MANO hand model, based on 1538 triangular faces in the mesh """
    triangular_faces = triangular_faces.tolist()

    graph = [[], []]
    for face in triangular_faces:
        # triangular face corner 1
        graph[0].append(face[0])
        graph[1].append(face[1])
        graph[0].append(face[1])
        graph[1].append(face[0])

        # triangular face corner 2
        graph[0].append(face[1])
        graph[1].append(face[2])
        graph[0].append(face[2])
        graph[1].append(face[1])

        # triangular face corner 3
        graph[0].append(face[2])
        graph[1].append(face[0])
        graph[0].append(face[0])
        graph[1].append(face[2])

    # graph vertices hand
    GRAPH_778_VERTICES_RHAND = torch.LongTensor(graph)
    torch.save(GRAPH_778_VERTICES_RHAND, 'GRAPH_VERTICES778_RHAND.pt')

    GRAPH_778_VERTICES_LHAND = torch.add(GRAPH_778_VERTICES_RHAND, 778)
    torch.save(GRAPH_778_VERTICES_LHAND, 'GRAPH_VERTICES778_LHAND.pt')

    GRAPH_778_VERTICES_2HANDS = torch.cat((GRAPH_778_VERTICES_RHAND, GRAPH_778_VERTICES_LHAND), -1)
    torch.save(GRAPH_778_VERTICES_2HANDS, 'GRAPH_VERTICES778_2HANDS.pt')


def compute_joint_graph():
    """ Computes a graph for 21 or 15 joints of the MANO hand model """

    # graph joints hand
    GRAPH_JOINTS21_HAND_R = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 4, 4, 5, 5, 6, 0, 7, 7, 8, 8, 9, 0, 10, 10, 11, 11, 12,    0, 13, 13, 14, 14, 15, 16, 15, 17, 3, 18, 6, 19, 12, 20, 9],
                                        [1, 0, 2, 1, 3, 2, 4, 0, 5, 4, 6, 5, 7, 0, 8, 7, 9, 8, 10, 0, 11, 10, 12, 11,    13, 0, 14, 13, 15, 14, 15, 16, 3, 17, 6, 18, 12, 19, 9, 20]],
                                         dtype=torch.long)
    torch.save(GRAPH_JOINTS21_HAND_R, 'GRAPH_JOINTS21_HAND_R.pt')

    GRAPH_JOINTS21_HAND_L = torch.add(GRAPH_JOINTS21_HAND_R, 21)
    torch.save(GRAPH_JOINTS21_HAND_L, 'GRAPH_JOINTS21_HAND_L.pt')

    GRAPH_JOINTS21_HAND_RL = torch.cat((GRAPH_JOINTS21_HAND_R, GRAPH_JOINTS21_HAND_L), -1)
    torch.save(GRAPH_JOINTS21_HAND_RL, 'GRAPH_JOINTS21_HAND_RL.pt')

    # graph fullpose hand
    GRAPH_JOINTS15_HAND_R = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14],
                                          [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10, 13, 12, 14, 13]],
                                         dtype=torch.long)
    torch.save(GRAPH_JOINTS15_HAND_R, 'GRAPH_JOINTS15_HAND_R.pt')

    GRAPH_JOINTS15_HAND_L = torch.add(GRAPH_JOINTS15_HAND_R, 15)
    torch.save(GRAPH_JOINTS15_HAND_L, 'GRAPH_JOINTS15_HAND_L.pt')

    GRAPH_JOINTS15_HAND_RL = torch.cat((GRAPH_JOINTS15_HAND_R, GRAPH_JOINTS15_HAND_L), -1)
    torch.save(GRAPH_JOINTS15_HAND_RL, 'GRAPH_JOINTS15_HAND_RL.pt')

    # graph joints arm
    GRAPH_JOINTS21_ARM_R = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 4, 4, 5, 5, 6, 0, 7, 7, 8, 8, 9, 0, 10, 10, 11, 11, 12,        0, 13, 13, 14, 14, 15, 16, 15, 17, 3, 18, 6, 19, 12, 20, 9,   0, 21, 21, 22, 22, 23, 23, 24, 24, 25],
                                         [1, 0, 2, 1, 3, 2, 4, 0, 5, 4, 6, 5, 7, 0, 8, 7, 9, 8, 10, 0, 11, 10, 12, 11,        13, 0, 14, 13, 15, 14, 15, 16, 3, 17, 6, 18, 12, 19, 9, 20,   21, 0, 22, 21, 23, 22, 24, 23, 25, 24]],
                                         dtype=torch.long)
    torch.save(GRAPH_JOINTS21_ARM_R, 'GRAPH_JOINTS21_ARM_R.pt')

    GRAPH_JOINTS21_ARM_L = torch.add(GRAPH_JOINTS21_ARM_R, 26)
    torch.save(GRAPH_JOINTS21_ARM_L, 'GRAPH_JOINTS21_ARM_L.pt')

    # remove the duplicated neck and set joint 25 as a neck
    GRAPH_JOINTS21_ARM_L = GRAPH_JOINTS21_ARM_L[:, :-2]
    connected_neck = torch.tensor([[50, 25], [25, 50]], dtype=torch.long)
    GRAPH_JOINTS21_ARM_L = torch.cat((GRAPH_JOINTS21_ARM_L, connected_neck), 1)
    GRAPH_JOINTS21_ARM_RL = torch.cat((GRAPH_JOINTS21_ARM_R, GRAPH_JOINTS21_ARM_L), -1)
    torch.save(GRAPH_JOINTS21_ARM_RL, 'GRAPH_JOINTS21_ARM_RL.pt')

    # graph fullpose arm
    GRAPH_JOINTS15_ARM_R = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14,    0, 15, 3, 15, 6, 15, 9, 15, 12, 15,     15, 16, 16, 17, 17, 18, 18, 19],
                                         [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10, 13, 12, 14, 13,    15, 0, 15, 3, 15, 6, 15, 9, 15, 12,     16, 15, 17, 16, 18, 17, 19, 18]],
                                         dtype=torch.long)
    torch.save(GRAPH_JOINTS15_ARM_R, 'GRAPH_JOINTS15_ARM_R.pt')

    GRAPH_JOINTS15_ARM_L = torch.add(GRAPH_JOINTS15_ARM_R, 20)
    torch.save(GRAPH_JOINTS15_ARM_L, 'GRAPH_JOINTS15_ARM_L.pt')

    # remove the duplicated neck and set joint 19 as a neck
    GRAPH_JOINTS15_ARM_L = GRAPH_JOINTS15_ARM_L[:, :-2]
    connected_neck = torch.tensor([[38, 19], [19, 38]], dtype=torch.long)
    GRAPH_JOINTS15_ARM_L = torch.cat((GRAPH_JOINTS15_ARM_L, connected_neck), 1)
    GRAPH_JOINTS15_ARM_RL = torch.cat((GRAPH_JOINTS15_ARM_R, GRAPH_JOINTS15_ARM_L), -1)
    torch.save(GRAPH_JOINTS15_ARM_RL, 'GRAPH_JOINTS15_ARM_RL.pt')


def load_graph_edges(graph_name):
    """ Loads graph edges with a given name as a tensor  """
    root_folder = get_project_root()
    graph_edges = torch.load(root_folder + f'/graphs/graph_definitions/{graph_name}.pt')
    return graph_edges


def get_arm_indexes():
    """ Returns indexes for right and left arm in the SMPL-X model  """
    r_arm_indexes = [21, 19, 17, 14, 12]
    l_arm_indexes = [20, 18, 16, 13, 12]
    return r_arm_indexes, l_arm_indexes


def get_arm_and_hand_indexes():
    """ Returns indexes for right and left arm in the SMPL-X model  """
    r_arm_indexes, l_arm_indexes = get_arm_indexes()
    r_arm_indexes = list(range(40, 55)) + r_arm_indexes
    l_arm_indexes = list(range(25, 40)) + l_arm_indexes
    return r_arm_indexes, l_arm_indexes
