import torch
import torch_mlir

class kernel_bicg(torch.nn.Module):
    def __init__(self):
        super(kernel_bicg, self).__init__()
    
    def forward(self, A, s, q, p, r):
        # s := A^T * r (transpose matrix-vector multiply)
        # q := A * p (matrix-vector multiply)
        
        # Initialize s to zeros
        # s[j] += r[i] * A[i][j] for all i,j
        # This is equivalent to: s = A^T @ r
        s_out = A.t() @ r
        
        # Initialize q to zeros
        # q[i] += A[i][j] * p[j] for all i,j
        # This is equivalent to: q = A @ p
        q_out = A @ p
        
        return s_out, q_out