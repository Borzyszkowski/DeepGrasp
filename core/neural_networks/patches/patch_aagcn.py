""" 
Overwrites the method tsagcn.AAGCN at runtime.
"""

import torch
import torch.nn as nn
import torch_geometric_temporal.nn.attention.tsagcn as tsagcn

from torch_geometric_temporal.nn.attention.tsagcn import GraphAAGCN
from torch_geometric_temporal.nn.attention.tsagcn import UnitGCN
from torch_geometric_temporal.nn.attention.tsagcn import UnitTCN


class AAGCN(nn.Module):
    r"""Two-Stream Adaptive Graph Convolutional Network.

    For details see this paper: `"Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition." <https://arxiv.org/abs/1805.07694>`_.
    This implementation is based on the authors Github Repo https://github.com/lshiwjx/2s-AGCN.
    It's used by the author for classifying actions from sequences of 3D body joint coordinates.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        edge_index (PyTorch LongTensor): Graph edge indices.
        num_nodes (int): Number of nodes in the network.
        stride (int, optional): Time strides during temporal convolution. (default: :obj:`1`)
        residual (bool, optional): Applying residual connection. (default: :obj:`True`)
        adaptive (bool, optional): Adaptive node connection weights. (default: :obj:`True`)
        attention (bool, optional): Applying spatial-temporal-channel-attention.
        (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_index: torch.LongTensor,
        num_nodes: int,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True,
        attention: bool = True,
        kernel_size: int = 9,
    ):
        super(AAGCN, self).__init__()
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        self.graph = GraphAAGCN(self.edge_index, self.num_nodes)
        self.A = self.graph.A

        self.gcn1 = UnitGCN(
            in_channels, out_channels, self.A, adaptive=adaptive, attention=attention
        )
        self.tcn1 = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = UnitTCN(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            )

    def forward(self, x):
        """
        Making a forward pass.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods,
            with shape (B, F_in, T_in, N_nodes).

        Return types:
            * **X** (PyTorch FloatTensor)* - Sequence of node features,
            with shape (B, out_channels, T_in//stride, N_nodes).
        """
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


tsagcn.AAGCN = AAGCN
print("WARNING: Patched torch_geometric_temporal AAGCN to use kernel_size==9")
