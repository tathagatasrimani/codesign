import sys

# Assuming filename is passed as a command-line argument
filename = sys.argv[1] if len(sys.argv) > 1 else "pytorch_test"

code = ""
with open(f"aten_code_{filename}.txt","r") as code_file:
   code = code_file.read()

prim_to_native = {"torch.ops.prims.collapse_view.default": "nativePrims.collapse_view",
                  "torch.ops.prims.transpose.default": "nativePrims.transpose",
                  "torch.ops.aten.mm.default": "nativePrims.transpose.mm",
                  "torch.ops.prims.mul.default": "nativePrims.mul",
                  "torch.ops.prims.add.default": "nativePrims.add",
                  "torch.ops.prims.le.default": "nativePrims.le",
                  "torch.ops.prims.where.default": "nativePrims.where",
                  "torch.ops.prims.broadcast_in_dim": "nativePrims.broadcast_in_dim"
                  }

for prim_func, native_func in prim_to_native.items():
   code = code.replace(prim_func, native_func)

with open(f"native_code_{filename}.txt","w") as native_file:
   native_file.write(code)