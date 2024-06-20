def collapse_view(matrix, from_dim, to_dim):
    def flatten(nested_list):
        """flatten a list of lists """
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(item)
        return result

    def collapse(matrix, current_dim, from_dim, to_dim):
        """ Recursively collapse the specified dimensions """
        if current_dim >= to_dim:
            return matrix
        elif current_dim >= from_dim:
            return flatten([collapse(sub_matrix, current_dim + 1, from_dim, to_dim) for sub_matrix in matrix])
        else:
            return [collapse(sub_matrix, current_dim + 1, from_dim, to_dim) for sub_matrix in matrix]

    return collapse(matrix, 0, from_dim, to_dim)

def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed

def mm(list1, list2):
    '''/* copied from pytorch comment。
    Matrix product of two Tensors.
    The behavior depends on the dimensionality of the Tensors as follows:
    - If both Tensors are 1-dimensional, (1d) the dot product (scalar) is returned.
    - If the arguments are 2D - 1D or 1D - 2D, the matrix-vector product is returned.
    - If both arguments are 2D, the matrix-matrix product is returned.
    - If one of the arguments is ND with N >= 3 and the other is 1D or 2D, and some
    conditions on the strides apply (see should_fold) we fold the first N-1 dimensions
    of the ND argument to form a matrix, call mm or mv, reshape it back to ND and return it
    - Otherwise, we return bmm, after broadcasting and folding the batched dimensions if
    there's more than one
    TODO: Folding is not implemented
    */'''
    def dot_product(vec1, vec2):
        return sum(x * y for x, y in zip(vec1, vec2))
    
    def matrix_vector_product(mat, vec):
        return [dot_product(row, vec) for row in mat]
    
    def matrix_matrix_product(mat1, mat2):
        mat2_t = list(zip(*mat2))  # Transpose mat2
        return [[dot_product(row, col) for col in mat2_t] for row in mat1]
    
    dim1 = len(list1)
    dim2 = len(list2)

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
            list1 = [list1]  # Treat as single row matrix
            return matrix_vector_product(list2, list1)[0]  # We get a single row matrix, extract the row
        else:
            # Both arguments are 1D
            return dot_product(list1, list2)

def le(input, value):
    if isinstance(input, list):
        return [le(elem, value) for elem in input]
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
    if isinstance(matrix1, (int, float)):
        return matrix1 + matrix2
    else:
        return [add(sub1, sub2) for sub1, sub2 in zip(matrix1, matrix2)]

def where(condition, x, y):
    if isinstance(condition, bool):
        return x if condition else y
    else:
        return [where(c, x_elem, y_elem) for c, x_elem, y_elem in zip(condition, x, y)]

def broadcast_in_dim(a, shape, broadcast_dimensions):
    # Calculate the number of dimensions for output matrix
    ndim_a = len(shape)
    ndim_result = max(ndim_a, len(broadcast_dimensions))
    
    # Expand dimensions of 'a' to match 'ndim_result'
    for _ in range(ndim_result - ndim_a):
        a = [a]
    
    # Adjust shape of 'a' to match 'ndim_result'
    while len(shape) < ndim_result:
        shape.insert(0, 1)
    
    # Broadcast 'a' along 'broadcast_dimensions'
    for dim in broadcast_dimensions:
        if shape[dim] == 1:
            shape[dim] = len(a)
        else:
            assert shape[dim] == len(a), "Cannot broadcast dimensions"
    
    # Perform actual broadcasting
    def recursive_broadcast(arr, idx):
        if idx >= ndim_result:
            return arr
        if isinstance(arr, list):
            return [recursive_broadcast(subarr, idx + 1) for subarr in arr]
        else:
            return [arr] * shape[idx]
    
    return recursive_broadcast(a, 0)