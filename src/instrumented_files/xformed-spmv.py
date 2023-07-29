import sys
from instrument_lib import *
import sys
from instrument_lib import *
import numpy as np
print(1, 4)
A_0 = instrument_read(np, 'np').random.randint(0, 512, size=(400, 400))
write_instrument_read(A_0, 'A_0')
print('malloc', sys.getsizeof(A_0), 'A_0')
print(1, 5)
B_0 = instrument_read(np, 'np').random.randint(512, high=None, size=(400, 400))
write_instrument_read(B_0, 'B_0')
print('malloc', sys.getsizeof(B_0), 'B_0')


def csr(matrix1, matrix2):
    print('enter scope 1')
    print(1, 8)
    matrix1_1 = instrument_read(matrix1, 'matrix1')
    write_instrument_read(matrix1_1, 'matrix1_1')
    print('malloc', sys.getsizeof(matrix1_1), 'matrix1_1')
    matrix2_1 = instrument_read(matrix2, 'matrix2')
    write_instrument_read(matrix2_1, 'matrix2_1')
    print('malloc', sys.getsizeof(matrix2_1), 'matrix2_1')
    print(3, 9)
    rowNum_1 = int(instrument_read_sub(instrument_read(matrix1_1,
        'matrix1_1').shape, 'matrix1_1.shape', 0, None, None, False))
    write_instrument_read(rowNum_1, 'rowNum_1')
    print('malloc', sys.getsizeof(rowNum_1), 'rowNum_1')
    print(3, 10)
    columnNum_1 = int(instrument_read_sub(instrument_read(matrix1_1,
        'matrix1_1').shape, 'matrix1_1.shape', 1, None, None, False))
    write_instrument_read(columnNum_1, 'columnNum_1')
    print('malloc', sys.getsizeof(columnNum_1), 'columnNum_1')
    print(3, 11)
    Value_1 = instrument_read(np, 'np').array([], dtype=int)
    write_instrument_read(Value_1, 'Value_1')
    print('malloc', sys.getsizeof(Value_1), 'Value_1')
    print(3, 12)
    Column_1 = instrument_read(np, 'np').array([], dtype=int)
    write_instrument_read(Column_1, 'Column_1')
    print('malloc', sys.getsizeof(Column_1), 'Column_1')
    print(3, 13)
    RowPtr_1 = instrument_read(np, 'np').array([], dtype=int)
    write_instrument_read(RowPtr_1, 'RowPtr_1')
    print('malloc', sys.getsizeof(RowPtr_1), 'RowPtr_1')
    print(3, 14)
    Result_1 = instrument_read(np, 'np').array([], dtype=int)
    write_instrument_read(Result_1, 'Result_1')
    print('malloc', sys.getsizeof(Result_1), 'Result_1')
    for i_1 in range(instrument_read(rowNum_1, 'rowNum_1')):
        print(5, 18)
        flag_1 = 1
        write_instrument_read(flag_1, 'flag_1')
        print('malloc', sys.getsizeof(flag_1), 'flag_1')
        for j_1 in range(instrument_read(columnNum_1, 'columnNum_1')):
            if instrument_read_sub(instrument_read_sub(instrument_read(
                matrix1_1, 'matrix1_1'), 'matrix1_1', instrument_read(i_1,
                'i_1'), None, None, False), 'matrix1_1[i_1]',
                instrument_read(j_1, 'j_1'), None, None, False) != 0:
                print(10, 21)
                Value_1 = instrument_read(np, 'np').append(instrument_read(
                    Value_1, 'Value_1'), instrument_read(np, 'np').array(
                    instrument_read_sub(instrument_read_sub(instrument_read
                    (matrix1_1, 'matrix1_1'), 'matrix1_1', instrument_read(
                    i_1, 'i_1'), None, None, False), 'matrix1_1[i_1]',
                    instrument_read(j_1, 'j_1'), None, None, False)))
                write_instrument_read(Value_1, 'Value_1')
                print('malloc', sys.getsizeof(Value_1), 'Value_1')
                print(10, 22)
                Column_1 = instrument_read(np, 'np').append(instrument_read
                    (Column_1, 'Column_1'), instrument_read(j_1, 'j_1'))
                write_instrument_read(Column_1, 'Column_1')
                print('malloc', sys.getsizeof(Column_1), 'Column_1')
                if instrument_read(flag_1, 'flag_1') == 1:
                    print(12, 24)
                    RowPtr_1 = instrument_read(np, 'np').append(instrument_read
                        (RowPtr_1, 'RowPtr_1'), len(instrument_read(Value_1,
                        'Value_1')) - 1)
                    write_instrument_read(RowPtr_1, 'RowPtr_1')
                    print('malloc', sys.getsizeof(RowPtr_1), 'RowPtr_1')
                    print(12, 25)
                    flag_1 = 0
                    write_instrument_read(flag_1, 'flag_1')
                    print('malloc', sys.getsizeof(flag_1), 'flag_1')
    print(6, 26)
    RowPtr_1 = instrument_read(np, 'np').append(instrument_read(RowPtr_1,
        'RowPtr_1'), 8)
    write_instrument_read(RowPtr_1, 'RowPtr_1')
    print('malloc', sys.getsizeof(RowPtr_1), 'RowPtr_1')
    for i_1 in range(instrument_read(rowNum_1, 'rowNum_1')):
        print(15, 31)
        start_1 = int(instrument_read_sub(instrument_read(RowPtr_1,
            'RowPtr_1'), 'RowPtr_1', instrument_read(i_1, 'i_1'), None,
            None, False))
        write_instrument_read(start_1, 'start_1')
        print('malloc', sys.getsizeof(start_1), 'start_1')
        print(15, 32)
        end_1 = int(instrument_read_sub(instrument_read(RowPtr_1,
            'RowPtr_1'), 'RowPtr_1', instrument_read(i_1, 'i_1') + 1, None,
            None, False))
        write_instrument_read(end_1, 'end_1')
        print('malloc', sys.getsizeof(end_1), 'end_1')
        print(15, 33)
        temp_1 = 0
        write_instrument_read(temp_1, 'temp_1')
        print('malloc', sys.getsizeof(temp_1), 'temp_1')
        for j_1 in range(instrument_read(start_1, 'start_1'),
            instrument_read(end_1, 'end_1')):
            print(18, 35)
            k_1 = int(instrument_read_sub(instrument_read(Column_1,
                'Column_1'), 'Column_1', instrument_read(j_1, 'j_1'), None,
                None, False))
            write_instrument_read(k_1, 'k_1')
            print('malloc', sys.getsizeof(k_1), 'k_1')
            print(18, 36)
            temp_1 += instrument_read_sub(instrument_read(Value_1,
                'Value_1'), 'Value_1', instrument_read(j_1, 'j_1'), None,
                None, False) * instrument_read_sub(instrument_read_sub(
                instrument_read(matrix2_1, 'matrix2_1'), 'matrix2_1',
                instrument_read(k_1, 'k_1'), None, None, False),
                'matrix2_1[k_1]', 0, None, None, False)
            write_instrument_read(temp_1, 'temp_1')
        print(19, 37)
        Result_1 = instrument_read(np, 'np').append(instrument_read(
            Result_1, 'Result_1'), instrument_read(temp_1, 'temp_1'))
        write_instrument_read(Result_1, 'Result_1')
        print('malloc', sys.getsizeof(Result_1), 'Result_1')
    print('exit scope 1')
    return instrument_read(Result_1, 'Result_1')
    print('exit scope 1')


if instrument_read(__name__, '__name__') == '__main__':
    print(22, 42)
    Result_0 = csr(instrument_read(A_0, 'A_0'), instrument_read(B_0, 'B_0'))
    write_instrument_read(Result_0, 'Result_0')
    print('malloc', sys.getsizeof(Result_0), 'Result_0')
    print('CSR result is:', Result_0)
