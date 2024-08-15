class nativeTensor():
    def __init__(self, data):
        self.data = data
        self.shape = self._get_shape_recursive(data)
        self.transpose_dim = None
        ''' Only support broadcasting one dimension'''
        self.broadcast_dim = None
        self.broadcast_length = None
    
    def _get_shape_recursive(self, data):
        if isinstance(data, list):
            return [len(data)] + self._get_shape_recursive(data[0])
        return []
    
    def set_transpose(self, dim0, dim1):
        if self.transpose_dim != None:
            # only allow transposing once
            self.realize()
        self.transpose_dim = (dim0, dim1)
        len0 = self.shape[dim0]
        self.shape[dim0] = self.shape[dim1]
        self.shape[dim1] = len0

    def set_broadcast(self, dim, length):
        self.broadcast_dim = dim
        self.broadcast_length = length
        self.shape.insert(dim, length)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        data = self.data
        if self.transpose_dim != None:
            list_indices = list(indices)
            indices0 = list_indices[self.transpose_dim[0]]
            list_indices[self.transpose_dim[0]] = list_indices[self.transpose_dim[1]]
            list_indices[self.transpose_dim[1]] = indices0
            indices = tuple(list_indices)
        if self.broadcast_dim != None:
            if len(indices) <= self.broadcast_dim:
                pass
            elif indices[self.broadcast_dim] >= self.broadcast_length:
                raise IndexError("Index out of range")
            else:
                list_indices = list(indices)
                list_indices.pop(self.broadcast_dim)
                indices = tuple(list_indices)
        for idx in indices:
            data = data[idx]
        return nativeTensor(data) if isinstance(data, list) else data
    
    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def realize(self):
        if len(self.shape)==3:
            self.data = [[[self[i,j,k] for k in range(self.shape[-1])] for j in range(self.shape[-2])] for i in range(self.shape[-3])]
        elif len(self.shape)==2:
            self.data = [[self[j,k] for k in range(self.shape[-1])] for j in range(self.shape[-2])]
        else:
            raise RuntimeError("Not Implemented for more than 3d matrix")
        self.broadcast_dim = None
        self.broadcast_length = None
        

class nativePrims:
    ''' Inputs are assumed to be nativeTensor
        Always create new Tensor after calculation
    '''

    def collapse_view(matrix: nativeTensor, from_dim, to_dim):
        def flatten(nested_list):
            """flatten a list of lists """
            result = []
            for item in nested_list:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result

        def collapse(matrix: nativeTensor, current_dim, from_dim, to_dim):
            """ Recursively collapse the specified dimensions """
            if current_dim >= to_dim:
                return matrix
            elif current_dim >= from_dim:
                return flatten([collapse(sub_matrix, current_dim + 1, from_dim, to_dim) for sub_matrix in matrix])
            else:
                return [collapse(sub_matrix, current_dim + 1, from_dim, to_dim) for sub_matrix in matrix]

        return collapse(matrix, 0, from_dim, to_dim)

    def transpose(matrix: nativeTensor, dims=[0,1]):
        # cpp implementation:
        # https://github.com/pytorch/pytorch/blob/4bdb4bbd864690d2d742d11574b661178bc2de0f/aten/src/ATen/native/vulkan/ops/Permute.cpp#L75
        assert len(dims)==2, "transpose take 2 dimensions"
        assert dims[0]<3 and dims[1]<3, "suppport up to 3d transpose"
        return matrix.set_transpose(*dims)

    def mm(list1, list2):
        # cpp implementation:
        # https://github.com/pytorch/pytorch/blob/4bdb4bbd864690d2d742d11574b661178bc2de0f/aten/src/ATen/native/LinearAlgebra.cpp#L2187
        def to_floats(nested_list):
            if isinstance(nested_list[0], list):
                return [[float(item) for item in sublist] for sublist in nested_list]
            else:
                return [float(item) for item in nested_list]
        
        def dot_product(vec1, vec2):
            vec1 = [float(x) for x in vec1]
            vec2 = [float(x) for x in vec2]
            return sum(x * y for x, y in zip(vec1, vec2))
        
        def matrix_vector_product(mat, vec):
            vec = [float(x) for x in vec]
            return [dot_product(row, vec) for row in mat]
        
        def vector_matrix_product(vec, mat):
            vec = [float(x) for x in vec]
            return [dot_product(vec, col) for col in zip(*mat)]
        
        def matrix_matrix_product(mat1, mat2):
            mat2_t = list(zip(*mat2))  # Transpose mat2
            return [[dot_product(row, col) for col in mat2_t] for row in mat1]
        
        list1 = to_floats(list1)
        list2 = to_floats(list2)
        
        if isinstance(list1[0], list):
            if isinstance(list2[0], list):
                # Both arguments are 2D (or higher)
                return matrix_matrix_product(list1, list2)
            else:
                # list1 is 2D and list2 is 1D
                return matrix_vector_product(list1, list2)
        else:
            if isinstance(list2[0], list):
                # list1 is 1D and list2 is 2D
                return vector_matrix_product(list1, list2)
            else:
                # Both arguments are 1D
                return dot_product(list1, list2)

    def le(input, value):
        # cpp implementation:
        # https://github.com/pytorch/pytorch/blob/4bdb4bbd864690d2d742d11574b661178bc2de0f/aten/src/ATen/native/BinaryOps.cpp#L1430
        if isinstance(input, list):
            return [nativePrims.le(elem, value) for elem in input]
        else:
            return input <= value

    def mul(input, other):
        def is_scalar(value):
            return isinstance(value, (int, float))
        
        def multiply(mat1, mat2):
            if is_scalar(mat2):
                # Multiply each element in mat1 by the scalar mat2
                if isinstance(mat1, list):
                    return [multiply(el, mat2) for el in mat1]
                else:
                    return mat1 * mat2
            elif is_scalar(mat1):
                # Multiply each element in mat2 by the scalar mat1
                if isinstance(mat2, list):
                    return [multiply(mat1, el) for el in mat2]
                else:
                    return mat1 * mat2
            else:
                # Both mat1 and mat2 are lists, perform element-wise multiplication
                return [multiply(el1, el2) for el1, el2 in zip(mat1, mat2)]

        return multiply(input, other)

    def add(matrix1, matrix2):
        def is_vector(mat):
            return isinstance(mat, list) and all(isinstance(i, (int, float)) for i in mat)
        
        def broadcast_vector_to_matrix(vec, shape):
            return [vec for _ in range(shape[0])]
        
        if is_vector(matrix1):
            matrix1 = broadcast_vector_to_matrix(matrix1, (len(matrix2), len(matrix2[0])))
        
        if is_vector(matrix2):
            matrix2 = broadcast_vector_to_matrix(matrix2, (len(matrix1), len(matrix1[0])))
        
        result = []
        for row1, row2 in zip(matrix1, matrix2):
            result.append([val1 + val2 for val1, val2 in zip(row1, row2)])
        
        return result

    def where(condition, x, y):
        if isinstance(x, (int, float)):
            x = [[x] * len(row) for row in condition]
        if isinstance(y, (int, float)):
            y = [[y] * len(row) for row in condition]
        
        result = []
        
        for cond_row, x_row, y_row in zip(condition, x, y):
            result_row = []
            for cond_val, x_val, y_val in zip(cond_row, x_row, y_row):
                result_row.append(x_val if cond_val else y_val)
            result.append(result_row)
        
        return result

    def broadcast_in_dim(input_tensor, target_shape, broadcast_dimensions):
        '''Only case broadcast_dimensions == [1] is properly implemented
            other cases are simulated'''
        def create_list(dimensions):
            if len(dimensions) == 1:
                return [None] * dimensions[0]
            return [create_list(dimensions[1:]) for _ in range(dimensions[0])]
        if broadcast_dimensions == [1] and len(target_shape)==2:
            return [input_tensor for _ in range(target_shape[1])]
        else:
            return create_list(target_shape)
