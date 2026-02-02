import torch
import torch_mlir

class syr2k(torch.nn.Module):
    def __init__(self):
        super(syr2k, self).__init__()
        # Register alpha and beta as buffers
        self.register_buffer('alpha', torch.tensor(1.0))
        self.register_buffer('beta', torch.tensor(1.0))
    
    def forward(self, C, A, B):
        # C := beta*C + alpha*(A*B^T + B*A^T)
        # where only the lower triangular part of C is updated
        
        # Scale C by beta
        C_scaled = torch.mul(C, self.beta)
        
        # Compute A * B^T + B * A^T
        # This is the symmetric rank-2k update
        update = A @ B.t() + B @ A.t()
        
        # Scale the update by alpha
        update_scaled = torch.mul(update, self.alpha)
        
        # Add to scaled C
        C_out = C_scaled + update_scaled
        
        # Extract lower triangular part (including diagonal)
        # since the original only updates C[i][j] for j <= i
        C_out = torch.tril(C_out)
        
        return C_out