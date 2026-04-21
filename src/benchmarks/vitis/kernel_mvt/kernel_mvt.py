import torch
import torch_mlir

class kernel_mvt(torch.nn.Module):
    def __init__(self):
        super(kernel_mvt, self).__init__()
    
    def forward(self, x1, x2, y_1, y_2, A):
        # x1 = x1 + A * y_1
        # First loop: x1[i] += A[i][j] * y_1[j] for all i,j
        # This is equivalent to: x1 = x1 + A @ y_1
        x1_out = x1 + A @ y_1
        
        # x2 = x2 + A^T * y_2
        # Second loop: x2[i] += A[j][i] * y_2[j] for all i,j
        # Note: A[j][i] means we're using the transpose of A
        # This is equivalent to: x2 = x2 + A^T @ y_2
        x2_out = x2 + A.t() @ y_2
        
        return x1_out, x2_out