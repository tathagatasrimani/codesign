import sys
from instrument_lib import *
import numpy as np
print(1, 4)
A_0 = np.random.randint(0, 512, size=(10, 10))
write_instrument_read(A_0, 'A_0')
print('malloc', sys.getsizeof(A_0), 'A_0')
print(1, 5)
B_0 = np.random.randint(512, high=None, size=(10, 10))
write_instrument_read(B_0, 'B_0')
print('malloc', sys.getsizeof(B_0), 'B_0')


def csr(matrix1_1, matrix2_1):
    print('enter scope 1')
    print(1, 8)
    print(3, 9)
    rowNum_1 = int(instrument_read_sub(instrument_read(matrix1_1,
        'matrix1_1').shape, "instrument_read(matrix1_1, 'matrix1_1').shape",
        0, None, None, False))
    write_instrument_read(rowNum_1, 'rowNum_1')
    print('malloc', sys.getsizeof(rowNum_1), 'rowNum_1')
    print(3, 10)
    columnNum_1 = int(instrument_read_sub(instrument_read(matrix1_1,
        'matrix1_1').shape, "instrument_read(matrix1_1, 'matrix1_1').shape",
        1, None, None, False))
    write_instrument_read(columnNum_1, 'columnNum_1')
    print('malloc', sys.getsizeof(columnNum_1), 'columnNum_1')
    print(3, 11)
    Value_1 = np.array([], dtype=int)
    write_instrument_read(Value_1, 'Value_1')
    print('malloc', sys.getsizeof(Value_1), 'Value_1')
    print(3, 12)
    Column_1 = np.array([], dtype=int)
    write_instrument_read(Column_1, 'Column_1')
    print('malloc', sys.getsizeof(Column_1), 'Column_1')
    print(3, 13)
    RowPtr_1 = np.array([], dtype=int)
    write_instrument_read(RowPtr_1, 'RowPtr_1')
    print('malloc', sys.getsizeof(RowPtr_1), 'RowPtr_1')
    print(3, 14)
    Result_1 = np.array([], dtype=int)
    write_instrument_read(Result_1, 'Result_1')
    print('malloc', sys.getsizeof(Result_1), 'Result_1')
    print('enter scope 2')
    for i_2 in range(instrument_read(rowNum_1, 'rowNum_1')):
        print(5, 18)
        flag_2 = 1
        write_instrument_read(flag_2, 'flag_2')
        print('malloc', sys.getsizeof(flag_2), 'flag_2')
        print('enter scope 3')
        for j_3 in range(instrument_read(columnNum_1, 'columnNum_1')):
            print('enter scope 4')
            if instrument_read_sub(instrument_read_sub(instrument_read(
                matrix1_1, 'matrix1_1'),
                "instrument_read(matrix1_1, 'matrix1_1')", instrument_read(
                i_2, 'i_2'), None, None, False),
                """instrument_read_sub(instrument_read(matrix1_1, 'matrix1_1'),
    "instrument_read(matrix1_1, 'matrix1_1')", instrument_read(i_2, 'i_2'),
    None, None, False)"""
                , instrument_read(j_3, 'j_3'), None, None, False) != 0:
                print(8, 20)
                print(10, 21)
                Value_1 = np.append(instrument_read(Value_1, 'Value_1'), np
                    .array(instrument_read_sub(instrument_read_sub(
                    instrument_read(matrix1_1, 'matrix1_1'),
                    "instrument_read(matrix1_1, 'matrix1_1')",
                    instrument_read(i_2, 'i_2'), None, None, False),
                    """instrument_read_sub(instrument_read(matrix1_1, 'matrix1_1'),
    "instrument_read(matrix1_1, 'matrix1_1')", instrument_read(i_2, 'i_2'),
    None, None, False)"""
                    , instrument_read(j_3, 'j_3'), None, None, False)))
                write_instrument_read(Value_1, 'Value_1')
                print('malloc', sys.getsizeof(Value_1), 'Value_1')
                print(10, 22)
                Column_1 = np.append(instrument_read(Column_1, 'Column_1'),
                    instrument_read(j_3, 'j_3'))
                write_instrument_read(Column_1, 'Column_1')
                print('malloc', sys.getsizeof(Column_1), 'Column_1')
                print('enter scope 5')
                if instrument_read(flag_2, 'flag_2') == 1:
                    print(10, 23)
                    print(12, 24)
                    RowPtr_1 = np.append(instrument_read(RowPtr_1,
                        'RowPtr_1'), len(instrument_read(Value_1, 'Value_1'
                        )) - 1)
                    write_instrument_read(RowPtr_1, 'RowPtr_1')
                    print('malloc', sys.getsizeof(RowPtr_1), 'RowPtr_1')
                    print(12, 25)
                    flag_2 = 0
                    write_instrument_read(flag_2, 'flag_2')
                    print('malloc', sys.getsizeof(flag_2), 'flag_2')
                else:
                    print(10, 23)
                print('exit scope 5')
            else:
                print(8, 20)
            print('exit scope 4')
        print('exit scope 3')
    print('exit scope 2')
    print(6, 26)
    RowPtr_1 = np.append(instrument_read(RowPtr_1, 'RowPtr_1'), 8)
    write_instrument_read(RowPtr_1, 'RowPtr_1')
    print('malloc', sys.getsizeof(RowPtr_1), 'RowPtr_1')
    print('enter scope 6')
    for i_6 in range(instrument_read(rowNum_1, 'rowNum_1')):
        print(15, 31)
        start_6 = int(instrument_read_sub(instrument_read(RowPtr_1,
            'RowPtr_1'), "instrument_read(RowPtr_1, 'RowPtr_1')",
            instrument_read(i_6, 'i_6'), None, None, False))
        write_instrument_read(start_6, 'start_6')
        print('malloc', sys.getsizeof(start_6), 'start_6')
        print(15, 32)
        end_6 = int(instrument_read_sub(instrument_read(RowPtr_1,
            'RowPtr_1'), "instrument_read(RowPtr_1, 'RowPtr_1')", 
            instrument_read(i_6, 'i_6') + 1, None, None, False))
        write_instrument_read(end_6, 'end_6')
        print('malloc', sys.getsizeof(end_6), 'end_6')
        print(15, 33)
        temp_6 = 0
        write_instrument_read(temp_6, 'temp_6')
        print('malloc', sys.getsizeof(temp_6), 'temp_6')
        print('enter scope 7')
        for j_7 in range(instrument_read(start_6, 'start_6'),
            instrument_read(end_6, 'end_6')):
            print(18, 35)
            k_7 = int(instrument_read_sub(instrument_read(Column_1,
                'Column_1'), "instrument_read(Column_1, 'Column_1')",
                instrument_read(j_7, 'j_7'), None, None, False))
            write_instrument_read(k_7, 'k_7')
            print('malloc', sys.getsizeof(k_7), 'k_7')
            print(18, 36)
            temp_6 += instrument_read_sub(instrument_read(Value_1,
                'Value_1'), "instrument_read(Value_1, 'Value_1')",
                instrument_read(j_7, 'j_7'), None, None, False
                ) * instrument_read_sub(instrument_read_sub(instrument_read
                (matrix2_1, 'matrix2_1'),
                "instrument_read(matrix2_1, 'matrix2_1')", instrument_read(
                k_7, 'k_7'), None, None, False),
                """instrument_read_sub(instrument_read(matrix2_1, 'matrix2_1'),
    "instrument_read(matrix2_1, 'matrix2_1')", instrument_read(k_7, 'k_7'),
    None, None, False)"""
                , 0, None, None, False)
            write_instrument_read(temp_6, 'temp_6')
        print('exit scope 7')
        print(19, 37)
        Result_1 = np.append(instrument_read(Result_1, 'Result_1'),
            instrument_read(temp_6, 'temp_6'))
        write_instrument_read(Result_1, 'Result_1')
        print('malloc', sys.getsizeof(Result_1), 'Result_1')
    print('exit scope 6')
    return instrument_read(Result_1, 'Result_1')
    print('exit scope 1')


print('enter scope 8')
if __name__ == '__main__':
    print(1, 41)
    print(22, 42)
    Result_8 = csr(instrument_read(A_0, 'A_0'), instrument_read(B_0, 'B_0'))
    write_instrument_read(Result_8, 'Result_8')
    print('malloc', sys.getsizeof(Result_8), 'Result_8')
    print('CSR result is:', instrument_read(Result_8, 'Result_8'))
else:
    print(1, 41)
print('exit scope 8')
