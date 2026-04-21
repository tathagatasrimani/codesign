import torch
import torch_mlir

class kernel_3mm(torch.nn.Module):
    def __init__(self):
        super(kernel_3mm, self).__init__()
    
    def forward(self, A, B, C, D):
        # G := (A*B) * (C*D)
        # E := A * B
        E = A @ B
        
        # F := C * D
        F = C @ D
        
        # G := E * F
        G = E @ F
        
        return G