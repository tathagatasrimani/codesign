import sys
from instrument_lib import *
import sys
from instrument_lib import *
import numpy as np
print(1, 4)
A__0 = instrument_read(np, 'np').random.randint(0, 512, size=(10, 10))
write_instrument_read(A__0, 'A__0')
print('malloc', sys.getsizeof(A__0), 'A__0')
print(1, 5)
B__0 = instrument_read(np, 'np').random.randint(512, high=None, size=(10, 10))
write_instrument_read(B__0, 'B__0')
print('malloc', sys.getsizeof(B__0), 'B__0')


def csr(matrix1, matrix2):
    print('enter scope 1')
    print(1, 8)
    matrix1__1 = instrument_read(matrix1, 'matrix1')
    write_instrument_read(matrix1__1, 'matrix1__1')
    print('malloc', sys.getsizeof(matrix1__1), 'matrix1__1')
    matrix2__1 = instrument_read(matrix2, 'matrix2')
    write_instrument_read(matrix2__1, 'matrix2__1')
    print('malloc', sys.getsizeof(matrix2__1), 'matrix2__1')
    print(3, 9)
    rowNum__1 = int(instrument_read_sub(instrument_read(matrix1__1,
        'matrix1__1').shape, 'matrix1__1.shape', 0, None, None, False))
    write_instrument_read(rowNum__1, 'rowNum__1')
    print('malloc', sys.getsizeof(rowNum__1), 'rowNum__1')
    print(3, 10)
    columnNum__1 = int(instrument_read_sub(instrument_read(matrix1__1,
        'matrix1__1').shape, 'matrix1__1.shape', 1, None, None, False))
    write_instrument_read(columnNum__1, 'columnNum__1')
    print('malloc', sys.getsizeof(columnNum__1), 'columnNum__1')
    print(3, 11)
    Value__1 = instrument_read(np, 'np').array([], dtype=int)
    write_instrument_read(Value__1, 'Value__1')
    print('malloc', sys.getsizeof(Value__1), 'Value__1')
    print(3, 12)
    Column__1 = instrument_read(np, 'np').array([], dtype=int)
    write_instrument_read(Column__1, 'Column__1')
    print('malloc', sys.getsizeof(Column__1), 'Column__1')
    print(3, 13)
    RowPtr__1 = instrument_read(np, 'np').array([], dtype=int)
    write_instrument_read(RowPtr__1, 'RowPtr__1')
    print('malloc', sys.getsizeof(RowPtr__1), 'RowPtr__1')
    print(3, 14)
    Result__1 = instrument_read(np, 'np').array([], dtype=int)
    write_instrument_read(Result__1, 'Result__1')
    print('malloc', sys.getsizeof(Result__1), 'Result__1')
    for i__1 in range(instrument_read(rowNum__1, 'rowNum__1')):
        print('enter scope 2')
        print(5, 18)
        flag__2 = 1
        write_instrument_read(flag__2, 'flag__2')
        print('malloc', sys.getsizeof(flag__2), 'flag__2')
        for j__2 in range(instrument_read(columnNum__1, 'columnNum__1')):
            print('enter scope 3')
            if instrument_read_sub(instrument_read_sub(instrument_read(
                matrix1__1, 'matrix1__1'), 'matrix1__1', instrument_read(
                i__1, 'i__1'), None, None, False), 'matrix1__1[i__1]',
                instrument_read(j__2, 'j__2'), None, None, False) != 0:
                print(10, 21)
                Value__1 = instrument_read(np, 'np').append(instrument_read
                    (Value__1, 'Value__1'), instrument_read(np, 'np').array
                    (instrument_read_sub(instrument_read_sub(
                    instrument_read(matrix1__1, 'matrix1__1'), 'matrix1__1',
                    instrument_read(i__1, 'i__1'), None, None, False),
                    'matrix1__1[i__1]', instrument_read(j__2, 'j__2'), None,
                    None, False)))
                write_instrument_read(Value__1, 'Value__1')
                print('malloc', sys.getsizeof(Value__1), 'Value__1')
                print(10, 22)
                Column__1 = instrument_read(np, 'np').append(instrument_read
                    (Column__1, 'Column__1'), instrument_read(j__2, 'j__2'))
                write_instrument_read(Column__1, 'Column__1')
                print('malloc', sys.getsizeof(Column__1), 'Column__1')
                if instrument_read(flag__2, 'flag__2') == 1:
                    print(12, 24)
                    RowPtr__1 = instrument_read(np, 'np').append(
                        instrument_read(RowPtr__1, 'RowPtr__1'), len(
                        instrument_read(Value__1, 'Value__1')) - 1)
                    write_instrument_read(RowPtr__1, 'RowPtr__1')
                    print('malloc', sys.getsizeof(RowPtr__1), 'RowPtr__1')
                    print(12, 25)
                    flag__2 = 0
                    write_instrument_read(flag__2, 'flag__2')
                    print('malloc', sys.getsizeof(flag__2), 'flag__2')
            print('exit scope 3')
        print('exit scope 2')
    print(6, 26)
    RowPtr__1 = instrument_read(np, 'np').append(instrument_read(RowPtr__1,
        'RowPtr__1'), 8)
    write_instrument_read(RowPtr__1, 'RowPtr__1')
    print('malloc', sys.getsizeof(RowPtr__1), 'RowPtr__1')
    for i__1 in range(instrument_read(rowNum__1, 'rowNum__1')):
        print('enter scope 4')
        print(15, 31)
        start__4 = int(instrument_read_sub(instrument_read(RowPtr__1,
            'RowPtr__1'), 'RowPtr__1', instrument_read(i__1, 'i__1'), None,
            None, False))
        write_instrument_read(start__4, 'start__4')
        print('malloc', sys.getsizeof(start__4), 'start__4')
        print(15, 32)
        end__4 = int(instrument_read_sub(instrument_read(RowPtr__1,
            'RowPtr__1'), 'RowPtr__1', instrument_read(i__1, 'i__1') + 1,
            None, None, False))
        write_instrument_read(end__4, 'end__4')
        print('malloc', sys.getsizeof(end__4), 'end__4')
        print(15, 33)
        temp__4 = 0
        write_instrument_read(temp__4, 'temp__4')
        print('malloc', sys.getsizeof(temp__4), 'temp__4')
        for j__4 in range(instrument_read(start__4, 'start__4'),
            instrument_read(end__4, 'end__4')):
            print('enter scope 5')
            print(18, 35)
            k__5 = int(instrument_read_sub(instrument_read(Column__1,
                'Column__1'), 'Column__1', instrument_read(j__4, 'j__4'),
                None, None, False))
            write_instrument_read(k__5, 'k__5')
            print('malloc', sys.getsizeof(k__5), 'k__5')
            print(18, 36)
            temp__4 += instrument_read_sub(instrument_read(Value__1,
                'Value__1'), 'Value__1', instrument_read(j__4, 'j__4'),
                None, None, False) * instrument_read_sub(instrument_read_sub
                (instrument_read(matrix2__1, 'matrix2__1'), 'matrix2__1',
                instrument_read(k__5, 'k__5'), None, None, False),
                'matrix2__1[k__5]', 0, None, None, False)
            write_instrument_read(temp__4, 'temp__4')
            print('exit scope 5')
        print(19, 37)
        Result__1 = instrument_read(np, 'np').append(instrument_read(
            Result__1, 'Result__1'), instrument_read(temp__4, 'temp__4'))
        write_instrument_read(Result__1, 'Result__1')
        print('malloc', sys.getsizeof(Result__1), 'Result__1')
        print('exit scope 4')
    print('exit scope 1')
    return instrument_read(Result__1, 'Result__1')
    print('exit scope 1')


if instrument_read(__name__, '__name__') == '__main__':
    print(22, 42)
    Result__0 = csr(instrument_read(A__0, 'A__0'), instrument_read(B__0,
        'B__0'))
    write_instrument_read(Result__0, 'Result__0')
    print('malloc', sys.getsizeof(Result__0), 'Result__0')
    print('CSR result is:', Result__0)
