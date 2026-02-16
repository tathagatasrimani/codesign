import torch
import torch_mlir

class kernel_atax(torch.nn.Module):
    def __init__(self):
        super(kernel_atax, self).__init__()
    
    def forward(self, A, x):
        # y := A^T * (A * x)
        # tmp := A * x
        tmp = A @ x
        
        # y := A^T * tmp
        y = A.t() @ tmp
        
        return y