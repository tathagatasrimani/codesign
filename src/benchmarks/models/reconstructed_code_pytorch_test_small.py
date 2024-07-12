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
primals_1=[[-0.08322545886039734, 0.08691057562828064, 0.10134124755859375, 0.19961169362068176, -0.09935501217842102, 0.21899709105491638, -0.04788854718208313, 0.048594772815704346, 0.07020124793052673, -0.1574471890926361, 0.0960766077041626, -0.023233920335769653, -0.20435193181037903, 0.11323109269142151, 0.008854925632476807, 0.15822675824165344], [0.08697506785392761, 0.033580929040908813, -0.08757498860359192, 0.23407787084579468, -0.12138175964355469, -0.2418065071105957, -0.01883593201637268, -0.183314710855484, -0.17972517013549805, -0.12593024969100952, 0.1616165041923523, -0.16686370968818665, -0.17436480522155762, -0.09547179937362671, -0.038448989391326904, -0.0724758505821228], [-0.1799396574497223, -0.008463680744171143, -0.08688154816627502, -0.08177018165588379, 0.060151100158691406, 0.1063413918018341, 0.06921210885047913, 0.07951465249061584, 0.010732412338256836, 0.19890612363815308, 0.11538684368133545, -0.02554607391357422, -0.24163928627967834, 0.21277737617492676, -0.15575134754180908, 0.14384984970092773], [0.11089718341827393, -0.21547752618789673, 0.13398078083992004, 0.11801645159721375, 0.17909115552902222, -0.1445329785346985, 0.08032643795013428, 0.23446553945541382, 0.04456344246864319, 0.09927138686180115, 0.054759681224823, 0.03431558609008789, -0.026470661163330078, -0.050542742013931274, 0.023796409368515015, 0.04707491397857666], [-0.2123783826828003, 0.025965481996536255, -0.1483553946018219, -0.18635308742523193, -0.12135565280914307, -0.06199970841407776, 0.1299780309200287, -0.009712010622024536, -0.049799323081970215, 0.17992404103279114, -0.06971174478530884, -0.14526861906051636, -0.13534250855445862, 0.025015681982040405, 0.1412067413330078, -0.23723071813583374], [-0.23265093564987183, -0.09473684430122375, 0.24889975786209106, -0.12799841165542603, 0.08264008164405823, -0.03792545199394226, 0.09050628542900085, -0.23240554332733154, -0.18310460448265076, 0.1080152690410614, -0.23188170790672302, -0.06674659252166748, -0.007579714059829712, -0.14605873823165894, -0.09607890248298645, -0.2152184545993805], [-0.22088253498077393, 0.18657585978507996, 0.11710250377655029, -0.005386471748352051, 0.10283041000366211, 0.04025498032569885, 0.04291984438896179, -0.06613990664482117, 0.0132540762424469, -0.16410529613494873, -0.23278948664665222, 0.15407249331474304, -0.07507306337356567, 0.17546242475509644, 0.038685142993927, 0.21053120493888855], [0.11564767360687256, 0.08276662230491638, -0.08822876214981079, -0.23609164357185364, -0.13873827457427979, -0.060539841651916504, -0.1428784728050232, -0.08423370122909546, 0.2401837706565857, 0.09564411640167236, 0.1579359769821167, 0.11075395345687866, -0.0319114625453949, 0.1750701367855072, 0.13243594765663147, -0.23440250754356384]]
primals_2=[-0.21543437242507935, 0.12783926725387573, -0.20471718907356262, -0.0693674385547638, -0.17587220668792725, 0.15850964188575745, -0.04347464442253113, 0.0942298173904419]
primals_3=[[-0.17309629917144775, -0.14704786241054535, 0.2923544645309448, -0.11260984092950821, -0.25419965386390686, 0.1173085868358612, 0.1613815873861313, -0.14354799687862396], [0.26736190915107727, 0.10813195258378983, 0.16761308908462524, -0.30838221311569214, -0.029679009690880775, -0.1898145228624344, 0.17796409130096436, -0.25117552280426025], [-0.22502703964710236, -0.180067777633667, 0.13265392184257507, 0.2215397208929062, 0.18856306374073029, -0.028890863060951233, -0.2049282193183899, 0.15491005778312683], [0.21022143959999084, 0.22280581295490265, -0.24079594016075134, -0.26682373881340027, 0.20515382289886475, 0.20919325947761536, 0.0809871032834053, -0.3118515610694885], [0.0107348021119833, -0.1001548171043396, 0.08860611915588379, 0.3158465325832367, 0.14012238383293152, -0.1179497241973877, -0.2686092257499695, -0.3182789087295532], [0.013843891210854053, -0.24130794405937195, 0.32127830386161804, 0.1708817034959793, 0.1031893938779831, -0.12051018327474594, -0.19615715742111206, 0.009695881977677345], [-0.21912264823913574, -0.06832855194807053, 0.14615565538406372, 0.26537635922431946, 0.0036106782499700785, 0.1504470854997635, 0.012831439264118671, -0.22421424090862274], [0.0013624811545014381, -0.13585986196994781, 0.2891373932361603, -0.1439872831106186, 0.007216214668005705, -0.2992459833621979, 0.20989374816417694, -0.3232818841934204]]
primals_4=[-0.2574911117553711, -0.045409686863422394, 0.18845701217651367, -0.3419632911682129, -0.31428420543670654, -0.35156071186065674, 0.2810169756412506, -0.1602572351694107]
primals_5=[[-0.14969822764396667, -0.20867173373699188, 0.2773570120334625, 0.24507948756217957, 0.26701056957244873, -0.19248604774475098, -0.22164644300937653, 0.09978042542934418]]
primals_6=[0.33406734466552734]
primals_7=None
primals_7=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
add_2, _, _, _, _, _ = forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7)