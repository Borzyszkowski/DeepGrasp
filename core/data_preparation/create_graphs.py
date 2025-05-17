""" Creates graphs for processing data with Graph Neural Networks """

import logging
import torch

from torch_geometric.data import Data

from graphs.graph_visualizers import (visualize_graph2D,
                                      visualize_graph3D,
                                      visualize_sequential_graph3D)
from graphs.graph_operations import load_graph_edges


def create_graph(node_features, label, graph_edges, visualize=False):
    """Creates a graph representation based on the given node features"""
    for features_tensor in node_features:
        graph = Data(x=features_tensor, y=label, edge_index=graph_edges)
        if visualize:
            visualize_graph2D(graph)
            visualize_graph3D(graph)
            logging.info(f'Number of nodes: {graph.num_nodes}')
            logging.info(f'Number of edges: {graph.num_edges}')
    return graph


def create_sequential_graph(node_features, index_label, graph_edges, visualize=False):
    """Creates a sequential graph representations based on the given list of node features"""
    sequential_graph = Data(x=node_features, y=index_label, edge_index=graph_edges)
    if visualize:
        logging.info(f'sequential_graph {sequential_graph}')
        logging.info(f'node_features shape: {node_features.shape}')
        logging.info(f'edge_index shape: {graph_edges.shape}')
        logging.info(f'label shape: {index_label.shape}')
        # Generate visualization for each sequential graph in the current batch
        for i, x in enumerate(node_features):
            y = index_label[i]
            visualize_sequential_graph3D(x, y, graph_edges)
    assert torch.max(graph_edges).item() == node_features.shape[2] - 1, "Number of edges and node features differ!"
    return sequential_graph


def select_hands(cfg):
    """ Select the hand(s) for processing, depending on the given configuration """
    assert True not in cfg.parse_hands.values() or True not in cfg.parse_arms.values(), "Arms and hands given at once!"

    if cfg.parse_hands["right"] and not cfg.parse_hands["left"]:
        return "HAND_R"
    elif cfg.parse_hands["left"] and not cfg.parse_hands["right"]:
        return "HAND_L"
    elif cfg.parse_hands["right"] and cfg.parse_hands["left"]:
        return "HAND_RL"
    elif cfg.parse_arms["right"] and not cfg.parse_arms["left"]:
        return "ARM_R"
    elif cfg.parse_arms["left"] and not cfg.parse_arms["right"]:
        return "ARM_L"
    elif cfg.parse_arms["right"] and cfg.parse_arms["left"]:
        return "ARM_RL"
    else:
        raise Exception("Wrong configuration for the hand parse has been given!")


def select_graphs(cfg, hand):
    """ Select the graph(s), depending on the desired input. Required also for the RNN to get the correct shapes. """
    graphs = {'hands': sum(cfg['parse_hands'].values()), 'arms': sum(cfg['parse_arms'].values())}
    for input_type, val in cfg.input_types.items():
        graphs[input_type] = {}
        if val is False:
            continue

        if input_type == 'fullpose':
            graph_name = "GRAPH_JOINTS15_" + hand
        elif input_type == 'contact' or input_type == 'verts':
            graph_name = "GRAPH_VERTICES778_" + hand.replace("ARM", "HAND")
        elif input_type == 'joints_15':
            graph_name = "GRAPH_JOINTS15_" + hand
        elif input_type == 'joints_21':
            graph_name = "GRAPH_JOINTS21_" + hand
        graph_edges = load_graph_edges(graph_name)
        num_nodes = max(graph_edges[0]).item() + 1

        graphs[input_type]['name'] = graph_name
        graphs[input_type]['edges'] = graph_edges
        graphs[input_type]['num_nodes'] = num_nodes
        graphs[input_type]['hand'] = hand
        logging.info(f'Selected graph: {graph_name}')

    multimodalities = sum(cfg.input_types.values())
    return graphs, multimodalities


def prepare_inputs(batch, input_type, cfg, selected_input):
    """ Prepares inputs for the training such as graphs for the GNNs or tensors for the RNNs """
    hand = selected_input[input_type]['hand']
    index_labels = torch.squeeze(batch['labels'], 1)
    graph_edges = selected_input[input_type]['edges'] if cfg.parse_graph else None

    # Select arm or hand
    r_part = 'rhand' if cfg.parse_hands["right"] else 'rarm'
    l_part = 'lhand' if cfg.parse_hands["left"] else 'larm'
    input_rh = batch['features'][r_part][input_type]
    input_lh = batch['features'][l_part][input_type]

    # Format input for the fullpose
    if input_type == 'fullpose':

        # Adding edge lengths as additional features
        edges_rh = batch['features'][r_part]["edges_15"]
        edges_lh = batch['features'][l_part]["edges_15"]

        input_rh = torch.cat((input_rh, edges_rh), axis=3).float()
        input_lh = torch.cat((input_lh, edges_lh), axis=3).float()
        input_lh = input_lh[:, :, :-1] if hand == "ARM_RL" else input_lh  # avoid duplication of the neck joint

    # Format input for the contact (touch)
    if input_type == 'contact':
        input_rh = torch.unsqueeze(input_rh, 3).float() if cfg.parse_graph else input_rh
        input_lh = torch.unsqueeze(input_lh, 3).float() if cfg.parse_graph else input_lh

    # Format input for the joints
    if input_type == 'joints_15' or input_type == 'joints_21':
        input_rh = input_rh.view(*input_rh.size()[:-1], 3).float() if cfg.parse_graph else input_rh
        input_lh = input_lh.view(*input_lh.size()[:-1], 3).float() if cfg.parse_graph else input_lh
        input_lh = input_lh[:, :, :-1] if hand == "ARM_RL" else input_lh  # avoid duplication of the neck joint

    # Format input for the verts
    if input_type == 'verts':
        input_rh = input_rh.float()
        input_lh = input_lh.float()

    # Generate graphs for GCNs or tensors for RNNs
    if hand in ["HAND_R", "ARM_R"] and hand not in ["HAND_L", "ARM_L"]:
        return create_sequential_graph(input_rh, index_labels, graph_edges) if cfg.parse_graph else input_rh
    if hand in ["HAND_L", "ARM_L"] and hand not in ["HAND_R", "ARM_R"]:
        return create_sequential_graph(input_lh, index_labels, graph_edges) if cfg.parse_graph else input_lh
    if hand in ["HAND_RL", "ARM_RL"]:
        input_2h = torch.cat((input_rh, input_lh), axis=2).float()
        return create_sequential_graph(input_2h, index_labels, graph_edges) if cfg.parse_graph else input_2h
