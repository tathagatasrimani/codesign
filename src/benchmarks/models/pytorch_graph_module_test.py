import torch
import torch.fx
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def relu_decomposition(x):
    return (x > 0) * x

def transform(m : nn.Module) -> nn.Module:
    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)

    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.aten.relu:
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(relu_decomposition, args=node.args)
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()

    return gm

def read_weights_from_file(x, y, z):
    W_q = torch.zeros((x, y, z))
    W_k = torch.zeros((x, y, z))
    return W_q, W_k

if __name__ == "__main__":
    model = NeuralNetwork()
    transformed_model = transform(model)
    print(transformed_model._code)