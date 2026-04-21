import torch
import torch_mlir

class AttentionHead(torch.nn.Module):
    def __init__(self):
        super(AttentionHead, self).__init__()
    
    def forward(self, query, key, value):
        """
        Scaled dot-product attention for a single head
        
        Args:
            query: [SEQ_LEN, HEAD_DIM] tensor
            key: [SEQ_LEN, HEAD_DIM] tensor
            value: [SEQ_LEN, HEAD_DIM] tensor
        
        Returns:
            attn_out: [SEQ_LEN, HEAD_DIM] tensor
        """
        # Get head dimension for scaling
        head_dim = query.size(-1)
        scale = torch.rsqrt(torch.tensor(float(head_dim), dtype=query.dtype, device=query.device))
        
        # Compute similarity: Q @ K^T / sqrt(d)
        # key.t() transposes key, then matmul with query
        similarity = torch.matmul(query, key.t())
        similarity = torch.mul(similarity, scale)
        
        # Softmax over sequence dimension (dim=-1)
        attention = torch.softmax(similarity, dim=-1)
        
        # Apply attention to values: attention @ V
        attn_out = torch.matmul(attention, value)
        
        return attn_out