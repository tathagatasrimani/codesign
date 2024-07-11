


def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7):
    transpose = torch.ops.prims.transpose.default(primals_1, [1, 0]);  primals_1 = None
    mm = torch.ops.aten.mm.default(primals_7, transpose);  transpose = None
    mul = torch.ops.prims.mul.default(mm, 1.0);  mm = None
    mul_1 = torch.ops.prims.mul.default(primals_2, 1.0);  primals_2 = None
    broadcast_in_dim = torch.ops.prims.broadcast_in_dim.default(mul_1, [1, 8], [1]);  mul_1 = None
    add = torch.ops.prims.add.default(mul, broadcast_in_dim);  mul = broadcast_in_dim = None
    le = torch.ops.prims.le.default(add, 0.0)
    where = torch.ops.prims.where.default(le, 0.0, add);  le = add = None
    transpose_1 = torch.ops.prims.transpose.default(primals_3, [1, 0]);  primals_3 = None
    mm_1 = torch.ops.aten.mm.default(where, transpose_1)
    mul_2 = torch.ops.prims.mul.default(mm_1, 1.0);  mm_1 = None
    mul_3 = torch.ops.prims.mul.default(primals_4, 1.0);  primals_4 = None
    broadcast_in_dim_1 = torch.ops.prims.broadcast_in_dim.default(mul_3, [1, 8], [1]);  mul_3 = None
    add_1 = torch.ops.prims.add.default(mul_2, broadcast_in_dim_1);  mul_2 = broadcast_in_dim_1 = None
    le_1 = torch.ops.prims.le.default(add_1, 0.0)
    where_1 = torch.ops.prims.where.default(le_1, 0.0, add_1);  le_1 = add_1 = None
    transpose_2 = torch.ops.prims.transpose.default(primals_5, [1, 0]);  primals_5 = None
    mm_2 = torch.ops.aten.mm.default(where_1, transpose_2)
    mul_4 = torch.ops.prims.mul.default(mm_2, 1.0);  mm_2 = None
    mul_5 = torch.ops.prims.mul.default(primals_6, 1.0);  primals_6 = None
    broadcast_in_dim_2 = torch.ops.prims.broadcast_in_dim.default(mul_5, [1, 1], [1]);  mul_5 = None
    add_2 = torch.ops.prims.add.default(mul_4, broadcast_in_dim_2);  mul_4 = broadcast_in_dim_2 = None
    return [add_2, primals_7, where, transpose_1, where_1, transpose_2]
    