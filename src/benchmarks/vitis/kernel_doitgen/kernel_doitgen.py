import torch
import torch_mlir

class kernel_doitgen(torch.nn.Module):
    def __init__(self):
        super(kernel_doitgen, self).__init__()
    
    def forward(self, A, C4):
        """
        Doitgen kernel - applies matrix multiplication along the innermost dimension
        
        Args:
            A: [NR, NQ, NP] 3D tensor
            C4: [NP, NP] 2D matrix
        
        Returns:
            A_out: [NR, NQ, NP] 3D tensor
        """
        nr, nq, np = A.size()
        
        # Reshape A to [NR*NQ, NP] for batch matrix multiplication
        A_reshaped = A.view(nr * nq, np)
        
        # Matrix multiply: [NR*NQ, NP] @ [NP, NP] = [NR*NQ, NP]
        # This computes sum[p] = A[r][q][s] * C4[s][p] for all r,q simultaneously
        result = torch.matmul(A_reshaped, C4)
        
        # Reshape back to [NR, NQ, NP]
        A_out = result.view(nr, nq, np)
        
        return A_out
# ```

# **Explanation:**

# The C kernel performs:
# ```
# For each (r, q) pair:
#     sum[p] = Σ_s A[r][q][s] * C4[s][p]
#     A[r][q][p] = sum[p]