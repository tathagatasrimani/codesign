import torch
assert torch.__version__ >= '2.0.*'

from torch._decomp import core_aten_decompositions
from torch._functorch.aot_autograd import aot_module_simplified

# Backends can further finetune the decompositions if needed
# Available decompositions can be found in
# torch/_decomp/decompositions.py and torch/_refs/__init__.py
decompositions = core_aten_decompositions()
decompositions.update(
    torch._decomp.get_decompositions([
        torch.ops.aten.addmm,
        torch.ops.aten.permute,
        torch.ops.aten.mul,
        torch.ops.aten.relu,
        torch.ops.aten.le,
        torch.ops.aten.scalar_tensor,
        torch.ops.aten.expand,
        torch.ops.aten.alias,
        torch.ops.aten.view,
        torch.ops.aten.as_strided
    ])
)

graph = None
def get_graph_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        global graph
        if not graph:
          graph = gm
        return gm

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm,
        sample_inputs,
        decompositions=decompositions,
        fw_compiler=my_compiler
    )

filename = "pytorch_test"
from benchmarks.models.pytorch_test import NeuralNetwork
model = NeuralNetwork()

torch._dynamo.reset()
fn = torch.compile(backend=get_graph_backend, dynamic=True)(model)
input = torch.rand(1, 28, 28, device='cpu')
out = fn(input)

with open(f"aten_code_{filename}.txt","w") as code_file:
   code_file.write(graph.code)