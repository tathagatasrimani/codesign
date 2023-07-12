import sys
from instrument_lib import *
import numpy as np
print(1, 4)
A__0 = np.random.randint(0, 512, size=(10, 10))
print(1, 5)
B__0 = np.random.randint(512, high=None, size=(10, 10))


def csr(matrix1, matrix2):
    print('enter scope 1')
    print(1, 8)
    matrix1__1 = matrix1
    matrix2__1 = matrix2
    print(3, 9)
    rowNum__1 = int(matrix1__1.shape[0])
    print(3, 10)
    columnNum__1 = int(matrix1__1.shape[1])
    print(3, 11)
    Value__1 = np.array([], dtype=int)
    print(3, 12)
    Column__1 = np.array([], dtype=int)
    print(3, 13)
    RowPtr__1 = np.array([], dtype=int)
    print(3, 14)
    Result__1 = np.array([], dtype=int)
    for i__1 in range(rowNum__1):
        print('enter scope 2')
        print(5, 18)
        flag__2 = 1
        for j__2 in range(columnNum__1):
            print('enter scope 3')
            if matrix1__1[i__1][j__2] != 0:
                print(10, 21)
                Value__1 = np.append(Value__1, np.array(matrix1__1[i__1][j__2])
                    )
                print(10, 22)
                Column__1 = np.append(Column__1, j__2)
                if flag__2 == 1:
                    print(12, 24)
                    RowPtr__1 = np.append(RowPtr__1, len(Value__1) - 1)
                    print(12, 25)
                    flag__2 = 0
            print('exit scope 3')
        print('exit scope 2')
    print(6, 26)
    RowPtr__1 = np.append(RowPtr__1, 8)
    for i__1 in range(rowNum__1):
        print('enter scope 4')
        print(15, 31)
        start__4 = int(RowPtr__1[i__1])
        print(15, 32)
        end__4 = int(RowPtr__1[i__1 + 1])
        print(15, 33)
        temp__4 = 0
        for j__4 in range(start__4, end__4):
            print('enter scope 5')
            print(18, 35)
            k__5 = int(Column__1[j__4])
            print(18, 36)
            temp__4 += Value__1[j__4] * matrix2__1[k__5][0]
            print('exit scope 5')
        print(19, 37)
        Result__1 = np.append(Result__1, temp__4)
        print('exit scope 4')
    print('exit scope 1')
    return Result__1
    print('exit scope 1')


if __name__ == '__main__':
    print(22, 42)
    Result__0 = csr(A__0, B__0)
    print('CSR result is:', Result__0)
