


def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8):
    collapse_view = nativePrims.collapse_view(primals_8, 1, 2);  primals_8 = None
    transpose = nativePrims.transpose(primals_1, [1, 0]);  primals_1 = None
    mm = nativePrims.mm(collapse_view, transpose);  transpose = None
    mul = nativePrims.mul(mm, 1.0);  mm = None
    mul_1 = nativePrims.mul(primals_2, 1.0);  primals_2 = None
    broadcast_in_dim = nativePrims.broadcast_in_dim(mul_1, [1, 512], [1]);  mul_1 = None
    add = nativePrims.add(mul, broadcast_in_dim);  mul = broadcast_in_dim = None
    le = nativePrims.le(add, 0.0)
    where = nativePrims.where(le, 0.0, add);  le = add = None
    transpose_1 = nativePrims.transpose(primals_3, [1, 0]);  primals_3 = None
    mm_1 = nativePrims.mm(where, transpose_1)
    mul_2 = nativePrims.mul(mm_1, 1.0);  mm_1 = None
    mul_3 = nativePrims.mul(primals_4, 1.0);  primals_4 = None
    broadcast_in_dim_1 = nativePrims.broadcast_in_dim(mul_3, [1, 512], [1]);  mul_3 = None
    add_1 = nativePrims.add(mul_2, broadcast_in_dim_1);  mul_2 = broadcast_in_dim_1 = None
    le_1 = nativePrims.le(add_1, 0.0)
    where_1 = nativePrims.where(le_1, 0.0, add_1);  le_1 = add_1 = None
    transpose_2 = nativePrims.transpose(primals_5, [1, 0]);  primals_5 = None
    mm_2 = nativePrims.mm(where_1, transpose_2)
    mul_4 = nativePrims.mul(mm_2, 1.0);  mm_2 = None
    mul_5 = nativePrims.mul(primals_6, 1.0);  primals_6 = None
    broadcast_in_dim_2 = nativePrims.broadcast_in_dim(mul_5, [1, 10], [1]);  mul_5 = None
    add_2 = nativePrims.add(mul_4, broadcast_in_dim_2);  mul_4 = broadcast_in_dim_2 = None
    return [add_2, collapse_view, where, transpose_1, where_1, transpose_2, primals_7]
    