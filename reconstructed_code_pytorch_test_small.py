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
        def get_shape(tensor):
            if isinstance(tensor, list):
                return [len(tensor)] + get_shape(tensor[0]) if tensor else []
            return []

        def can_broadcast(input_shape, target_shape, broadcast_dimensions):
            input_shape = [1] * (len(target_shape) - len(input_shape)) + input_shape
            for i, dim in enumerate(broadcast_dimensions):
                if input_shape[dim] != target_shape[dim] and input_shape[dim] != 1:
                    return False
            return True

        def broadcast(tensor, input_shape, target_shape, broadcast_dimensions):
            if not input_shape:
                return tensor
            if len(input_shape) == 1:
                if input_shape[0] == 1:
                    return [tensor] * target_shape[0]
                if input_shape[0] == target_shape[0]:
                    return tensor
                raise ValueError("Shape mismatch in broadcasting")
            
            if input_shape[0] == target_shape[0]:
                return [broadcast(t, input_shape[1:], target_shape[1:], broadcast_dimensions[1:]) for t in tensor]
            elif input_shape[0] == 1:
                return [broadcast(tensor[0], input_shape[1:], target_shape[1:], broadcast_dimensions[1:]) for _ in range(target_shape[0])]
            else:
                raise ValueError("Shape mismatch in broadcasting")

        input_shape = get_shape(input_tensor)
        adjusted_input_shape = [1] * (len(target_shape) - len(input_shape)) + input_shape
        if not can_broadcast(adjusted_input_shape, target_shape, broadcast_dimensions):
            raise ValueError(f"Cannot broadcast input shape {input_shape} to the target shape {target_shape} with the given broadcast dimensions {broadcast_dimensions}.")

        return broadcast(input_tensor, adjusted_input_shape, target_shape, broadcast_dimensions)







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
primals_1=[[0.219013512134552, 0.13252854347229004, -0.08501735329627991, -0.24549055099487305, -0.16954416036605835, -0.24922290444374084, -0.17965662479400635, 0.061369508504867554, -0.1927925944328308, -0.005758613348007202, -0.03151577711105347, 0.01520615816116333, -0.157996267080307, -0.2264346182346344, 0.18282687664031982, 0.14409086108207703], [0.14762276411056519, 0.12336403131484985, -0.13430002331733704, 0.1613500416278839, 0.23218899965286255, 0.0511283278465271, -0.0992499589920044, 0.23351722955703735, -0.05960431694984436, -0.14966782927513123, 0.24380609393119812, 0.10257920622825623, 0.06156912446022034, 0.23114222288131714, -0.06236529350280762, -0.15685361623764038], [0.23871642351150513, -0.055674999952316284, 0.10008475184440613, -0.06168070435523987, -0.02723178267478943, 0.09756508469581604, 0.0689619779586792, 0.13121065497398376, 0.1557016372680664, -0.17283254861831665, 0.12976613640785217, -0.17935869097709656, -0.09089535474777222, -0.21289688348770142, 0.24424272775650024, 0.15678289532661438], [-0.22243747115135193, -0.09466055035591125, -0.11507588624954224, -0.22936919331550598, -0.23223614692687988, 0.09267738461494446, 0.20350179076194763, -0.23928049206733704, -0.2420787811279297, -0.24233588576316833, -0.13781386613845825, 0.23906269669532776, -0.12099999189376831, 0.23362544178962708, -0.0010851621627807617, -0.024483829736709595], [0.2234111726284027, -0.22308406233787537, 0.12170058488845825, -0.11841505765914917, 0.03714275360107422, -0.14537116885185242, -0.22024095058441162, -0.08577033877372742, -0.2105122208595276, 0.06813859939575195, 0.15313389897346497, -0.16338661313056946, 0.18095171451568604, -0.11746940016746521, -0.1999828815460205, 0.001497119665145874], [0.1965220868587494, 0.16263169050216675, -0.14320746064186096, 0.09059438109397888, 0.20005172491073608, 0.026412904262542725, 0.048930853605270386, 0.17884668707847595, 0.0887863039970398, 0.15134891867637634, -0.02134588360786438, 0.0925217866897583, -0.23890173435211182, -0.04487800598144531, -0.06427401304244995, 0.02795863151550293], [-0.2169772982597351, 0.10091215372085571, 0.2486676573753357, -0.15617015957832336, 0.03297901153564453, 0.17553332448005676, 0.007033169269561768, -0.13855376839637756, -0.013027429580688477, -0.17375242710113525, 0.1779451072216034, -0.10350722074508667, 0.21398383378982544, -0.11879780888557434, 0.10130366683006287, -0.17931124567985535], [-0.18918165564537048, 0.14089038968086243, -0.11132490634918213, -0.020960181951522827, -0.0753915011882782, 0.20490458607673645, 0.06761690974235535, -0.14330875873565674, 0.10843059420585632, -0.07120111584663391, -0.10195755958557129, -0.13734248280525208, 0.11895444989204407, 0.1942748725414276, 0.030447453260421753, -0.11125189065933228]]
primals_2=[0.146086186170578, -0.16634663939476013, 0.10998806357383728, -0.10629567503929138, 0.04006505012512207, -0.09766897559165955, -0.12998655438423157, 0.18480971455574036]
primals_3=[[0.18489611148834229, 0.07519911974668503, -0.14214450120925903, -0.3077404797077179, 0.050626035779714584, 0.1877720057964325, -0.22084715962409973, -0.0414523109793663], [-0.20051813125610352, -0.22267714142799377, -0.2597843110561371, 0.0014901860849931836, -0.22211910784244537, -0.16381262242794037, -0.15700501203536987, 0.007209681905806065], [-0.23843908309936523, -0.16559286415576935, 0.23687401413917542, -0.020627932623028755, 0.09414476156234741, 0.14859727025032043, -0.1673269122838974, 0.3214702010154724], [0.24679230153560638, 0.19052128493785858, 0.13673321902751923, -0.013514723628759384, -0.19957628846168518, 0.314281702041626, 0.11232586205005646, 0.144520103931427], [0.3411034941673279, -0.3395100235939026, -0.20886114239692688, 0.08015605807304382, -0.1496896743774414, 0.09742395579814911, 0.1414756327867508, 0.30043256282806396], [-0.278759628534317, 0.07695862650871277, -0.09971117973327637, -0.16578863561153412, 0.2728080749511719, 0.15871596336364746, 0.3068148195743561, 0.25540268421173096], [-0.2842768728733063, -0.027515780180692673, 0.174796000123024, 0.33119258284568787, 0.33574920892715454, 0.11665961146354675, 0.11076967418193817, -0.014173057861626148], [-0.22527073323726654, -0.2757136821746826, -0.1972203105688095, 0.2891424596309662, -0.04775566607713699, -0.18768206238746643, 0.3056654632091522, -0.28452998399734497]]
primals_4=[-0.08800729364156723, -0.06782607734203339, 0.3044576942920685, -0.026791317388415337, -0.3027428388595581, -0.236751988530159, 0.13250800967216492, -0.1583186537027359]
primals_5=[[-0.3434229791164398, 0.32022547721862793, -0.2170349806547165, 0.045054178684949875, -0.001711077755317092, -0.31014373898506165, -0.3051157295703888, -0.3009248375892639]]
primals_6=[-0.06589267402887344]
primals_7=None
primals_7=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
add_2, _, _, _, _, _ = forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7)