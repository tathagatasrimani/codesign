'''
import torch
a=torch.tensor([range(8)]).squeeze()
print(f"a={a.size()}")

b = torch.ops.prims.broadcast_in_dim.default(a, [1, 8], [1])
print(f"b={b.size()}")

c = torch.ops.prims.broadcast_in_dim.default(b, [1, 8], [1])
print(f"c={c.size()}")

d = torch.ops.prims.broadcast_in_dim.default(c, [512, 512], [0, 1])
print(f"d={d.size()}")
'''

def shape(arr):
        def get_shape_recursive(arr, current_shape):
            if isinstance(arr, list):
                current_shape.append(len(arr))
                if isinstance(arr[0], list):
                    get_shape_recursive(arr[0], current_shape)
            return current_shape
        
        current_shape = get_shape_recursive(arr, [])
        return tuple(current_shape)

print(shape([[1,2]]))