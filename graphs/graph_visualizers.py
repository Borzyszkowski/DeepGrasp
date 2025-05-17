""" Visualizes 2D and 3D graphs for GCN """

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import os

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from tools.utils import get_project_root, makepath


def visualize_graph2D(graph_data):
    """ Visualize a graph for GCN in 2D """
    g = to_networkx(graph_data, to_undirected=True)
    nx.draw(g, with_labels=True)
    # plt.show()


def visualize_graph3D(graph_data, string_label=False):
    """ Visualize a graph for GCN in 3D """
    connections = graph_data.edge_index
    x = graph_data.x[:, 0]
    y = graph_data.x[:, 1]
    z = graph_data.x[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if string_label:
        ax.set_title(f"Plot of a 3D hand graph(s) for label: {string_label}")

    # Plot connections between the joints
    for point in range(len(connections[0])):
        start = connections[0][point]
        end = connections[1][point]
        ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], color='g')

    ax.scatter(x, y, z, color='g', marker="o")
    # plt.show()


def visualize_sequential_graph3D(graph_data, label_data, edge_index, string_label=None):
    """ Visualize a sequential 3D graph as a gif """
    root_folder = get_project_root()
    makepath(root_folder + '/plots', isfile=False)

    filenames = []
    for i in range(graph_data.shape[0]):
        # Get a single graph from a sequence
        single_graph = Data(x=graph_data[i], y=label_data, edge_index=edge_index)
        visualize_graph3D(single_graph, string_label)

        # Create file name and append it to a list
        filename = root_folder + f'/plots/{i}.png'
        filenames.append(filename)

        # Save frame
        plt.savefig(filename)
        plt.close()

    # Build gif
    with imageio.get_writer(root_folder + '/plots/sequential_graph3D.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)
