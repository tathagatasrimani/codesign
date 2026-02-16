import torch
import torch_mlir

class kernel_2mm(torch.nn.Module):
    def __init__(self):
        super(kernel_2mm, self).__init__()
        # Register alpha and beta as buffers instead of plain attributes
        # This ensures they're properly tracked as tensors in the MLIR conversion
        self.register_buffer('alpha', torch.tensor(0.1))
        self.register_buffer('beta', torch.tensor(0.5))
    
    def forward(self, A, B, C, D):
        # D := alpha*A*B*C + beta*D
        # First compute tmp = alpha * A * B
        tmp = torch.mul(A @ B, self.alpha)
        
        # Then compute D = tmp * C + beta * D
        D_scaled = torch.mul(D, self.beta)
        D_out = tmp @ C + D_scaled
        
        return D_out
import torch
import torch_mlir

class kernel_2mm(torch.nn.Module):
    def __init__(self):
        super(kernel_2mm, self).__init__()
        # Register alpha and beta as buffers instead of plain attributes
        # This ensures they're properly tracked as tensors in the MLIR conversion
        self.register_buffer('alpha', torch.tensor(0.1))
        self.register_buffer('beta', torch.tensor(0.5))
    
    def forward(self, A, B, C, D):
        # D := alpha*A*B*C + beta*D
        # First compute tmp = alpha * A * B
        tmp = torch.mul(A @ B, self.alpha)
        
        # Then compute D = tmp * C + beta * D
        D_scaled = torch.mul(D, self.beta)
        D_out = tmp @ C + D_scaled
        
        return D_out