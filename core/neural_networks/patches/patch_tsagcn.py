""" 
Overwrites the method tsagcn.UnitGCN._adaptive_forward at runtime.
This method contains a bug - open PR: 
https://github.com/benedekrozemberczki/pytorch_geometric_temporal/issues/291
https://github.com/benedekrozemberczki/pytorch_geometric_temporal/pull/292
"""

import torch
import torch_geometric_temporal.nn.attention.tsagcn as tsagcn


def patched_forward(self, x, y):
    N, C, T, V = x.size()

    A = self.PA
    for i in range(self.num_subset):
        A1 = (
            self.conv_a[i](x)
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(N, V, self.inter_c * T)
        )
        A2 = self.conv_b[i](x).reshape(N, self.inter_c * T, V)
        A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
        A1 = A[i] + A1 * self.alpha
        A2 = x.view(N, C * T, V)
        z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
        y = z + y if y is not None else z

    return y


tsagcn.UnitGCN._adaptive_forward = patched_forward
print("WARNING: Patched torch_geometric_temporal TSAGCN to use reshape() instead of view()")
