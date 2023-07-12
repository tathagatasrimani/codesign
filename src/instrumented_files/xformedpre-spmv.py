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
        print(5, 18)
        flag__1 = 1
        for j__1 in range(columnNum__1):
            if matrix1__1[i__1][j__1] != 0:
                print(10, 21)
                Value__1 = np.append(Value__1, np.array(matrix1__1[i__1][j__1])
                    )
                print(10, 22)
                Column__1 = np.append(Column__1, j__1)
                if flag__1 == 1:
                    print(12, 24)
                    RowPtr__1 = np.append(RowPtr__1, len(Value__1) - 1)
                    print(12, 25)
                    flag__1 = 0
    print(6, 26)
    RowPtr__1 = np.append(RowPtr__1, 8)
    for i__1 in range(rowNum__1):
        print(15, 31)
        start__1 = int(RowPtr__1[i__1])
        print(15, 32)
        end__1 = int(RowPtr__1[i__1 + 1])
        print(15, 33)
        temp__1 = 0
        for j__1 in range(start__1, end__1):
            print(18, 35)
            k__1 = int(Column__1[j__1])
            print(18, 36)
            temp__1 += Value__1[j__1] * matrix2__1[k__1][0]
        print(19, 37)
        Result__1 = np.append(Result__1, temp__1)
    print('exit scope 1')
    return Result__1
    print('exit scope 1')


if __name__ == '__main__':
    print(22, 42)
    Result__0 = csr(A__0, B__0)
    print('CSR result is:', Result__0)
