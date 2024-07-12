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
primals_1=[[0.02632749080657959, -0.13650906085968018, 0.22600233554840088, -0.14536413550376892, 0.22678884863853455, 0.22206854820251465, -0.14547640085220337, 0.19455286860466003, -0.16454026103019714, -0.21498516201972961, 0.01383596658706665, -0.02151450514793396, 0.16780051589012146, -0.03826180100440979, -0.060333818197250366, 0.23880741000175476], [0.01857692003250122, 0.1449163258075714, 0.2013406753540039, -0.038476407527923584, -0.06534320116043091, -0.0800119936466217, -0.10672619938850403, 0.21774378418922424, 0.22638851404190063, -0.16963329911231995, -0.03387343883514404, 0.07644274830818176, 0.08926838636398315, 0.16309231519699097, 0.01931557059288025, -0.07910224795341492], [-0.13537335395812988, -0.07941725850105286, -0.1462191343307495, -0.14873817563056946, -0.05321359634399414, 0.1586609184741974, -0.07151016592979431, -0.07696700096130371, -0.1897541582584381, -0.2119332253932953, 0.03906714916229248, -0.08667835593223572, 0.18165689706802368, 0.16996246576309204, 0.03532490134239197, -0.12676605582237244], [-0.17046865820884705, 0.18060371279716492, -0.14613786339759827, 0.24622541666030884, -0.2164730727672577, -0.13232830166816711, -0.154678612947464, 0.008006691932678223, -0.050525784492492676, -0.08284693956375122, -0.20727777481079102, -0.04679200053215027, 0.08833146095275879, 0.19830474257469177, -0.004647642374038696, 0.13750451803207397], [0.14823448657989502, 0.22761034965515137, -0.10937562584877014, -0.1513369381427765, 0.18354469537734985, 0.052858322858810425, -0.20552733540534973, 0.13575351238250732, -0.13797372579574585, 0.07331860065460205, -0.23935234546661377, 0.024072468280792236, -0.042278438806533813, -0.15570330619812012, -0.0780704915523529, 0.1210755705833435], [-0.04274338483810425, -0.0970592200756073, 0.15107831358909607, -0.1671350598335266, 0.143854022026062, -0.14529946446418762, -0.20402303338050842, 0.13953131437301636, 0.09578055143356323, 0.18018785119056702, -0.2244294285774231, 0.06351509690284729, 0.24700212478637695, 0.13778099417686462, 0.045388758182525635, -0.08095166087150574], [-0.008606761693954468, -0.10601526498794556, 0.034288644790649414, -0.17378395795822144, -0.1680806279182434, -0.23210591077804565, -0.11920303106307983, -0.10635581612586975, -0.14032253623008728, -0.02631285786628723, 0.13569706678390503, 0.19039928913116455, -0.04530438780784607, -0.13960957527160645, -0.21572217345237732, 0.002645641565322876], [0.16247594356536865, 0.11776798963546753, 0.17533516883850098, 0.18153396248817444, 0.20294395089149475, -0.14908257126808167, -0.05835515260696411, -0.00010591745376586914, 0.2182307243347168, -0.09294083714485168, -0.08013823628425598, -0.1661677062511444, 0.04987072944641113, 0.1096651554107666, 0.02574211359024048, 0.05064082145690918]]
primals_2=[0.08070462942123413, -0.111328125, 0.22219154238700867, -0.07846185564994812, 0.15514251589775085, 0.2090274691581726, 0.018453598022460938, 0.026767313480377197]
primals_3=[[0.29997003078460693, 0.13798801600933075, -0.13557562232017517, 0.18552245199680328, -0.058758899569511414, 0.0862826481461525, 0.3075096011161804, -0.1331745982170105], [-0.30645129084587097, -0.29792648553848267, -0.010367829352617264, -0.3057670295238495, -0.17913173139095306, -0.3228923976421356, -0.21458621323108673, -0.015764860436320305], [0.10403013974428177, 0.3403378129005432, -0.04013534635305405, 0.20018914341926575, 0.25561562180519104, -0.22162014245986938, -0.15653331577777863, -0.11353998631238937], [-0.019577380269765854, -0.042774371802806854, -0.3449150621891022, -0.2203403115272522, -0.08189613372087479, -0.17162996530532837, -0.03378424048423767, -0.06410754472017288], [0.20433597266674042, -0.06367933750152588, -0.028188318014144897, 0.29748034477233887, -0.15346716344356537, 0.0034707086160779, -0.2585086226463318, -0.07135448604822159], [-0.026351474225521088, -0.020863700658082962, 0.34051504731178284, 0.34969547390937805, 0.16127313673496246, 0.12313007563352585, 0.12585656344890594, -0.22861583530902863], [-0.3150356411933899, 0.2627057731151581, 0.19567520916461945, -0.0673815980553627, 0.027317436411976814, -0.16864003241062164, 0.3331206440925598, -0.14953145384788513], [0.14805205166339874, -0.10645361989736557, 0.3151494264602661, 0.26574307680130005, -0.1973007321357727, -0.29891058802604675, 0.13221605122089386, 0.32055261731147766]]
primals_4=[-0.060170821845531464, 0.054021552205085754, 0.00834975391626358, -0.19965171813964844, -0.11586518585681915, 0.15503524243831635, 0.16446883976459503, 0.24067185819149017]
primals_5=[[0.3366280496120453, 0.20214492082595825, 0.339470773935318, -0.1603575497865677, 0.07129712402820587, 0.2370472252368927, 0.15082873404026031, 0.11993290483951569]]
primals_6=[-0.05279790237545967]
primals_7=None
primals_7=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
add_2, _, _, _, _, _ = forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7)