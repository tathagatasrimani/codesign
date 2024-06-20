filename = "pytorch_test" # TODO: pass this from command arg
code = ""
with open(f"aten_code_{filename}.txt","r") as code_file:
   code = code_file.read()

prim_to_native = {"torch.ops.prims.collapse_view.default": "nativePrims.collapse_view",
                  "torch.ops.prims.transpose,default": "nativePrims.transpose",
                  "torch.ops.aten.mm.default": "nativePrims.transpose.mm",
                  "torch.ops.prims.mul.default": "nativePrims.mul",
                  "torch.ops.aten.add.Tensor": "nativePrims.add",
                  "torch.ops.prims.le.default": "nativePrims.le",
                  "torch.ops.aten.scalar_tensor.default": "nativePrims.scalar_tensor",
                  "torch.ops.aten.where.self": "nativePrims.where",
                  }

for prim_func, native_func in prim_to_native:
   code = code.replace(prim_func, native_func)

with open(f"native_code_{filename}.txt","w") as native_file:
   native_file.write(code)