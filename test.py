import torch
a=torch.tensor([range(512)]).squeeze()
print(f"a={a.size()}")

b = torch.ops.prims.broadcast_in_dim.default(a, [1, 512], [1])
print(f"b={b.size()}")

c = torch.ops.prims.broadcast_in_dim.default(b, [256, 512], [0, 1])
print(f"c={c.size()}")

d = torch.ops.prims.broadcast_in_dim.default(c, [512, 512], [0, 1])
print(f"d={d.size()}")