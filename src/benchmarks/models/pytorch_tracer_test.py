# https://pytorch.org/docs/stable/fx.html#torch.fx.Graph
import os
import torch
from torch import nn
import torch.fx as fx
from torch.fx import symbolic_trace
import torch.nn.functional as F

# os.environ["TORCH_COMPILE_DEBUG"] = '1'
device = "cpu"

def relu_decomposition(x):
    return (x > 0) * x

decomposition_rules = {}
decomposition_rules[F.relu] = relu_decomposition

def decompose(model: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    """
    Decompose `model` into smaller constituent operations.
    Currently,this only supports decomposing ReLU into its
    mathematical definition: (x > 0) * x
    """
    graph : fx.Graph = tracer_class().trace(model)
    new_graph = fx.Graph()
    env = {}
    tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)
    for node in graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            # By wrapping the arguments with proxies,
            # we can dispatch to the appropriate
            # decomposition rule and implicitly add it
            # to the Graph by symbolically tracing it.
            proxy_args = [
                fx.Proxy(env[x.name], tracer) if isinstance(x, fx.Node) else x for x in node.args]
            output_proxy = decomposition_rules[node.target](*proxy_args)

            # Operations on `Proxy` always yield new `Proxy`s, and the
            # return value of our decomposition rule is no exception.
            # We need to extract the underlying `Node` from the `Proxy`
            # to use it in subsequent iterations of this transform.
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            # Default case: we don't have a decomposition rule for this
            # node, so just copy the node over into the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    return fx.GraphModule(model, new_graph)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.relu(x)
        return x

def read_weights_from_file(x, y, z):
    W_q = torch.zeros((x, y, z))
    W_k = torch.zeros((x, y, z))
    return W_q, W_k

if __name__ == "__main__":
    model = NeuralNetwork()
    print("original:")
    symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
    print(symbolic_traced.code)
    
    print("decomposed:")
    decomposed_model = decompose(model)
    print(decomposed_model._code)