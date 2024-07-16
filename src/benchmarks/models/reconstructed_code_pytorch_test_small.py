class nativePrims:
    def shape(arr):
        def get_shape_recursive(arr, current_shape):
            if isinstance(arr, list):
                current_shape.append(len(arr))
                if isinstance(arr[0], list):
                    get_shape_recursive(arr[0], current_shape)
            return current_shape
        
        current_shape = get_shape_recursive(arr, [])
        return tuple(current_shape)

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

    def transpose(matrix, dims=[0,1]):
        assert len(dims)==2, "transpose take 2 dimensions"
        assert dims[0]<3 and dims[1]<3, "suppport up to 3d transpose"
        if dims[0]==dims[1]:
            return matrix
        if isinstance(matrix[0][0], list):
            # matrix is 3d
            if dims[0]==0 and dims[1]==1:
                return [[[matrix[i][j][k] for k in range(len(matrix[0][0]))] for i in range(len(matrix))] for j in range(len(matrix[0]))]
            elif dims[0]==0 and dims[1]==2:
                return [[[matrix[i][j][k] for i in range(len(matrix))] for j in range(len(matrix[0]))] for k in range(len(matrix[0][0]))]
            elif dims[0]==1 and dims[1]==2:
                return [[[matrix[i][j][k] for j in range(len(matrix[0]))] for k in range(len(matrix[0][0]))] for i in range(len(matrix))]
        else:
            #matrix is 2d
            rows = len(matrix)
            cols = len(matrix[0])
            
            transposed = [[0 for _ in range(rows)] for _ in range(cols)]
            
            for i in range(rows):
                for j in range(cols):
                    transposed[j][i] = matrix[i][j]

            return transposed

    def mm(list1, list2):
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





def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7):
    transpose = nativePrims.transpose(primals_1, [1, 0]);  primals_1 = None
    mm = nativePrims.mm(primals_7, transpose);  transpose = None
    mul = nativePrims.mul(mm, 1.0);  mm = None
    mul_1 = nativePrims.mul(primals_2, 1.0);  primals_2 = None
    broadcast_in_dim = nativePrims.broadcast_in_dim(mul_1, [1, 8], [1]);  mul_1 = None
    add = nativePrims.add(mul, broadcast_in_dim);  mul = broadcast_in_dim = None
    le = nativePrims.le(add, 0.0)
    where = nativePrims.where(le, 0.0, add);  le = add = None
    transpose_1 = nativePrims.transpose(primals_3, [1, 0]);  primals_3 = None
    mm_1 = nativePrims.mm(where, transpose_1)
    mul_2 = nativePrims.mul(mm_1, 1.0);  mm_1 = None
    mul_3 = nativePrims.mul(primals_4, 1.0);  primals_4 = None
    broadcast_in_dim_1 = nativePrims.broadcast_in_dim(mul_3, [1, 8], [1]);  mul_3 = None
    add_1 = nativePrims.add(mul_2, broadcast_in_dim_1);  mul_2 = broadcast_in_dim_1 = None
    le_1 = nativePrims.le(add_1, 0.0)
    where_1 = nativePrims.where(le_1, 0.0, add_1);  le_1 = add_1 = None
    transpose_2 = nativePrims.transpose(primals_5, [1, 0]);  primals_5 = None
    mm_2 = nativePrims.mm(where_1, transpose_2)
    mul_4 = nativePrims.mul(mm_2, 1.0);  mm_2 = None
    mul_5 = nativePrims.mul(primals_6, 1.0);  primals_6 = None
    broadcast_in_dim_2 = nativePrims.broadcast_in_dim(mul_5, [1, 1], [1]);  mul_5 = None
    add_2 = nativePrims.add(mul_4, broadcast_in_dim_2);  mul_4 = broadcast_in_dim_2 = None
    return [add_2, primals_7, where, transpose_1, where_1, transpose_2]
    
self=None
primals_1=[[0.06368255615234375, 0.1164344847202301, 0.04238376021385193, -0.04275655746459961, -0.17327478528022766, 0.2254578173160553, -0.23480671644210815, -0.1732141375541687, -0.11741667985916138, 0.13188838958740234, -0.1608545482158661, -0.17750197649002075, 0.15691232681274414, -0.24571329355239868, -0.10810968279838562, -0.2176588773727417], [0.22322198748588562, 0.18224984407424927, 0.05798956751823425, 0.1931847333908081, -0.09260714054107666, -0.03259757161140442, -0.17851606011390686, 0.19540420174598694, -0.07910078763961792, 0.1285725235939026, 0.10780960321426392, 0.099831223487854, 0.2025226652622223, -0.08494997024536133, -0.17778941988945007, -0.24069815874099731], [-0.1350482702255249, -0.14944863319396973, 0.14650383591651917, -0.12884479761123657, 0.2056930959224701, -0.08381828665733337, 0.030512064695358276, 0.006183892488479614, 0.07049620151519775, 0.07588276267051697, 0.02455994486808777, -0.07126975059509277, 0.13445422053337097, 0.06284117698669434, 0.008564352989196777, 0.13210606575012207], [0.0287514328956604, -0.04651728272438049, -0.036664336919784546, 0.1304612159729004, -0.22852224111557007, -0.010035067796707153, -0.2260744273662567, -0.06208300590515137, 0.15453201532363892, 0.08308145403862, -0.11697506904602051, 0.08729144930839539, 0.019569873809814453, -0.0996001660823822, 0.1065942645072937, 0.1739228069782257], [0.028292417526245117, -0.18432161211967468, -0.15094956755638123, -0.1557001769542694, -0.15090316534042358, -0.0032873153686523438, 0.24868470430374146, 0.02556547522544861, 0.19639387726783752, 0.11489111185073853, -0.22325900197029114, -0.10565215349197388, 0.10787475109100342, -0.1838003695011139, 0.21973946690559387, -0.19280242919921875], [0.15192991495132446, 0.06887423992156982, -0.06073084473609924, -0.2447134554386139, -0.016430824995040894, 0.041787683963775635, 0.18229854106903076, -0.08356598019599915, -0.19328990578651428, -0.17046838998794556, -0.2380700707435608, 0.23183518648147583, -0.031911253929138184, -0.05570521950721741, -0.10492914915084839, 0.1606304943561554], [-0.0029174387454986572, -0.18473803997039795, 0.11511734127998352, 0.1836862564086914, -0.13672778010368347, 0.04362618923187256, 0.07671463489532471, -0.21392172574996948, 0.14336875081062317, 0.22251072525978088, -0.24268817901611328, -0.03524613380432129, -0.10378298163414001, -0.04796549677848816, -0.011058539152145386, 0.005667775869369507], [0.12047550082206726, -0.09987658262252808, -0.041611433029174805, 0.01431986689567566, 0.17802977561950684, 0.02044445276260376, 0.030041426420211792, 0.19164758920669556, 0.009695827960968018, 0.07341006398200989, 0.09544387459754944, -0.20109733939170837, 0.167992502450943, -0.054093897342681885, -0.10008165240287781, -0.05521351099014282]]
primals_2=[0.1822817027568817, -0.12299063801765442, 0.14359897375106812, -0.11673596501350403, 0.07882261276245117, 0.19400835037231445, -0.16715794801712036, -0.17914500832557678]
primals_3=[[-0.24359866976737976, -0.33333441615104675, -0.16017475724220276, -0.3365650475025177, -0.21909651160240173, 0.14226529002189636, -0.3490573763847351, 0.050141219049692154], [-0.08105344325304031, -0.3306862413883209, 0.3231494426727295, 0.29168838262557983, 0.0824870690703392, -0.25858354568481445, -0.20344607532024384, 0.20835556089878082], [0.05236353725194931, -0.1890064924955368, 0.11956525593996048, -0.1722802072763443, 0.27254942059516907, 0.2280486673116684, 0.2636886239051819, -0.16893868148326874], [-0.04708009585738182, -0.23486052453517914, 0.2871924936771393, -0.1970585137605667, -0.1923215389251709, -0.18062107264995575, 0.023429011926054955, 0.2201317995786667], [0.021408239379525185, 0.22225946187973022, -0.003768686903640628, -0.18134431540966034, 0.12283138185739517, 0.22253906726837158, 0.2380685806274414, -0.0689227432012558], [0.24635553359985352, -0.3480229079723358, 0.2828602194786072, 0.02393013797700405, 0.07091502100229263, -0.0466061532497406, -0.1449916958808899, 0.23946629464626312], [0.2555159032344818, -0.12286927551031113, 0.13434576988220215, 0.20565178990364075, 0.03634525090456009, 0.29037249088287354, -0.2865239679813385, 0.305011123418808], [0.08055337518453598, -0.03807925432920456, 0.13862961530685425, -0.10858279466629028, -0.32366371154785156, 0.13371099531650543, -0.11821962893009186, 0.23416051268577576]]
primals_4=[-0.13475839793682098, 0.10219885408878326, -0.26846837997436523, -0.3393232226371765, -0.28847992420196533, 0.3386504650115967, -0.2548084259033203, 0.2795855402946472]
primals_5=[[0.26748260855674744, 0.05270674079656601, 0.2882717549800873, -0.26857122778892517, -0.07545343786478043, 0.07941216230392456, -0.1115168109536171, -0.13450635969638824]]
primals_6=[-0.1599685698747635]
primals_7=None
primals_7=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
add_2, _, _, _, _, _ = forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7)