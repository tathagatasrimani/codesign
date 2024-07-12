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
primals_1=[[-0.2054312825202942, 0.24622586369514465, -0.05507853627204895, 0.013924241065979004, 0.02829846739768982, 0.0780293345451355, -0.014918506145477295, -0.24990341067314148, -0.009665369987487793, -0.05234768986701965, 0.09062144160270691, -0.1689808964729309, -0.14807119965553284, 0.2228488028049469, -0.0012053251266479492, -0.14772510528564453], [0.17048847675323486, -0.12069082260131836, 0.12775376439094543, 0.2459459900856018, -0.1483127474784851, -0.14636856317520142, -0.11857736110687256, 0.20795932412147522, 0.20564004778862, 0.19805297255516052, -0.08329814672470093, -0.06938448548316956, -0.2427828013896942, -0.07424217462539673, 0.09530937671661377, -0.017236262559890747], [0.007236272096633911, 0.20565906167030334, 0.015515625476837158, -0.23644140362739563, 0.2000502645969391, -0.15492740273475647, -0.20681744813919067, -0.0031101107597351074, -0.15741804242134094, 0.1666553020477295, -0.0760321319103241, 0.0124092698097229, -0.20796671509742737, -0.21542224287986755, -0.03565606474876404, -0.15544405579566956], [-0.18376076221466064, 0.22868329286575317, -0.004898130893707275, -0.15390804409980774, 0.22756561636924744, -0.22404542565345764, -0.03292441368103027, 0.12582755088806152, -0.2031416893005371, -0.19975373148918152, 0.2273382544517517, -0.007100313901901245, -0.22726425528526306, 0.14717534184455872, -0.07243800163269043, -0.21185708045959473], [0.06931284070014954, -0.11072167754173279, 0.08446982502937317, -0.10959699749946594, -0.1729896068572998, 0.01808592677116394, -0.2180635929107666, 0.039285093545913696, -0.238925039768219, 0.05820772051811218, 0.10664442181587219, -0.20987892150878906, 0.0736781656742096, -0.23799148201942444, -0.10775703191757202, -0.02478855848312378], [-0.1494203507900238, 0.018401294946670532, 0.1776597797870636, 0.143819659948349, 0.027993619441986084, -0.18850678205490112, 0.20040854811668396, 0.18172332644462585, 0.1710011065006256, -0.02682909369468689, 0.004771500825881958, 0.08815914392471313, 0.12576702237129211, -0.2411557137966156, 0.15070968866348267, 0.17538344860076904], [0.006690531969070435, -0.16919907927513123, 0.07569536566734314, 0.15176784992218018, -0.21346959471702576, 0.194596529006958, -0.004566818475723267, -0.18490543961524963, 0.1544513702392578, 0.07843062281608582, -0.04435887932777405, -0.06617170572280884, -0.15576529502868652, 0.0752573013305664, 0.040387123823165894, -0.03792664408683777], [-0.21027982234954834, -0.024000883102416992, 0.07686722278594971, 0.0906425416469574, 0.05323082208633423, -0.17754527926445007, 0.059393107891082764, 0.13958263397216797, -0.07931077480316162, 0.13817477226257324, 0.10130777955055237, 0.015870362520217896, 0.1896381974220276, -0.1677969992160797, 0.20232868194580078, 0.08735677599906921]]
primals_2=[0.07745400071144104, -0.22840362787246704, 0.21899950504302979, 0.04816776514053345, 0.16179636120796204, 0.10651543736457825, -0.21380609273910522, 0.18567487597465515]
primals_3=[[-0.2856660783290863, -0.07788927108049393, 0.17861497402191162, -0.14748543500900269, 0.2494766265153885, -0.22428470849990845, -0.11890283226966858, -0.1898380070924759], [-0.07498670369386673, -0.2805900573730469, 0.08453667163848877, 0.2952497899532318, -0.19088876247406006, -0.0006115086143836379, 0.3231697976589203, 0.2592095732688904], [-0.2872841954231262, -0.022774891927838326, 0.14293618500232697, 0.27089813351631165, -0.08750414848327637, 0.15934285521507263, -0.16685132682323456, -0.23241356015205383], [0.3330875635147095, -0.055803269147872925, 0.17818839848041534, 0.09320670366287231, 0.3111436665058136, -0.10962963849306107, 0.07353942096233368, 0.023212840780615807], [0.1319970190525055, -0.1277679204940796, -0.1861158013343811, 0.2268243432044983, -0.1498865783214569, 0.12917865812778473, -0.14097829163074493, -0.2578526735305786], [0.005803494714200497, 0.2143574357032776, -0.021844416856765747, 0.3397916853427887, -0.24504926800727844, -0.1649651676416397, -0.34450796246528625, -0.33076825737953186], [-0.22410689294338226, -0.34669846296310425, 0.003945071250200272, 0.05749727785587311, -0.28726255893707275, 0.30772167444229126, -0.2665430009365082, -0.015514086000621319], [-0.3308350145816803, 0.20208558440208435, -0.060907043516635895, 0.05527331680059433, -0.2629771828651428, -0.2572697401046753, 0.18171735107898712, -0.2410566657781601]]
primals_4=[-0.18916866183280945, -0.06821471452713013, 0.17453216016292572, 0.2030186653137207, 0.12241008132696152, 0.19086435437202454, -0.0486568920314312, 0.133737251162529]
primals_5=[[-0.006717280484735966, 0.3447161912918091, -0.0906633511185646, 0.16268151998519897, -0.047285180538892746, 0.2753743827342987, 0.014819927513599396, 0.2999603748321533]]
primals_6=[-0.24863757193088531]
primals_7=None
primals_7=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
add_2, _, _, _, _, _ = forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7)