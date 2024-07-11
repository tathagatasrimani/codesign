import torch
import importlib.util
import sys
assert torch.__version__ >= '2.0.*'
import re
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
        torch.ops.aten.as_strided,
        torch.ops.aten.add,
        torch.ops.aten.where,
        torch.ops.aten.mm,
        torch.ops.aten.transpose,
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

prim_to_native = {"torch.ops.prims.collapse_view.default": "nativePrims.collapse_view",
                  "torch.ops.prims.transpose.default": "nativePrims.transpose",
                  "torch.ops.aten.mm.default": "nativePrims.mm",
                  "torch.ops.prims.mul.default": "nativePrims.mul",
                  "torch.ops.prims.add.default": "nativePrims.add",
                  "torch.ops.prims.le.default": "nativePrims.le",
                  "torch.ops.prims.where.default": "nativePrims.where",
                  "torch.ops.prims.broadcast_in_dim.default": "nativePrims.broadcast_in_dim"
                  }

# Assuming the network is named as "NeuralNetwork"
file_path = sys.argv[1] if len(sys.argv) > 1 else "src/benchmarks/models/pytorch_test.py"
print(f"decomposing {file_path}")
module_name = "pytorch_module"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
NeuralNetwork = module.NeuralNetwork
model = NeuralNetwork()

def find_first_layer_with_weights(module):
    for layer in module.children():
        if hasattr(layer, 'weight'):
            return layer.weight.shape
        else:
            # Recursively search in nested layers
            result = find_first_layer_with_weights(layer)
            if result is not None:
                return result
    return None

match = re.search(r'([^/]+)\.py$', file_path)
file_name = match.group(1)

# get forward function
torch._dynamo.reset()
fn = torch.compile(backend=get_graph_backend, dynamic=True)(model)
input_shape = find_first_layer_with_weights(model)
input = torch.rand(1, input_shape[1], device='cpu')
out = fn(input)
forward_code: str = graph.code

# save to file
with open(f"aten_code_{file_name}.py","w") as code_file:
   code_file.write(forward_code)

for prim_func, native_func in prim_to_native.items():
   forward_code = forward_code.replace(prim_func, native_func)

with open(f"native_code_{file_name}.py","w") as native_file:
   native_file.write(forward_code)

# parse function definition and return statement
code_lines = forward_code.split("\n")

def_line = ""
ret_line = ""
for line in code_lines:
    if line.startswith("def"):
        def_line = line
    elif line.startswith("    return "):
       ret_line = line

# Initialize arguments base on model
pattern = r'def\s+\w+\s*\(([^)]*)\)'
match = re.search(pattern, def_line)
arguments_str = match.group(1)
arguments = [arg.strip() for arg in arguments_str.split(',')]
argument_lines=[]
model_weights: dict = model.state_dict()
weights = list(model_weights.values())
cur_index = 0
num_weights = len(weights)
# Assume weights are passed in the same order as defined in model
for argument in arguments:
    if argument=="self":
        argument_lines.append(f"{argument}=None")
    elif cur_index<num_weights:
        argument_lines.append(f"{argument}={weights[cur_index].tolist()}")
        cur_index+=1
    else: # some arguments are not used, why are the generated?
        argument_lines.append(f"{argument}=None")
# Assume the input is the last argument, initiate the input with the same shape as first layer
argument_lines.append(f"{arguments[-1]}={[1]*weights[0].shape[1]}")

pattern = r'\s*return\s*\[\s*([^]]+)\s*\]'
match = re.search(pattern, ret_line)
values_str = match.group(1)
values = [arg.strip() for arg in values_str.split(',')]
invoke_line = ""
for index, value in enumerate(values):
    if index==0:
        invoke_line = invoke_line + value
    else:
        invoke_line = invoke_line + ", _"
pattern = r'(forward\(.*?\))\s*:'
match = re.search(pattern, def_line)
function_signature = match.group(1)
invoke_line = invoke_line + " = " + function_signature

with open(r"src/nativePrims.py", "r") as nativePrims:
    nativePrim_lines = nativePrims.read()
code_lines = [nativePrim_lines,"\n"] + code_lines + argument_lines + [invoke_line,]
code = "\n".join(code_lines)

with open(f"reconstructed_code_{file_name}.py","w") as reconstruct_file:
   reconstruct_file.write(code)


