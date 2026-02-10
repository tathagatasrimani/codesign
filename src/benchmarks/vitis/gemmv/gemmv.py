import torch
import torch_mlir

class gemmv(torch.nn.Module):
    def __init__(self):
        super(gemmv, self).__init__()
        # Register alpha and beta as buffers
        self.register_buffer('alpha', torch.tensor(0.1))
        self.register_buffer('beta', torch.tensor(0.5))
    
    def forward(self, A, x, y):
        """
        General Matrix-Vector multiplication: y = alpha * A @ x + y
        
        Args:
            A: [N, N] matrix
            x: [N] vector
            y: [N] vector (input, will be added to result)
        
        Returns:
            y_out: [N] vector
        """
        # Compute matrix-vector product: A @ x
        Ax = torch.matmul(A, x)
        
        # Scale by alpha
        scaled_Ax = torch.mul(Ax, self.alpha)
        
        # Add to y: y_out = alpha * A @ x + y
        y_out = scaled_Ax + y
        
        return y_out