import sys
from instrument_lib import *
import numpy as np
print(1, 4)
A = np.random.randint(0, 512, size=(10, 10))
print(1, 5)
B = np.random.randint(512, high=None, size=(10, 10))


def csr(matrix1, matrix2):
    print(1, 8)
    print(3, 9)
    rowNum = int(matrix1.shape[0])
    print(3, 10)
    columnNum = int(matrix1.shape[1])
    print(3, 11)
    Value = np.array([], dtype=int)
    print(3, 12)
    Column = np.array([], dtype=int)
    print(3, 13)
    RowPtr = np.array([], dtype=int)
    print(3, 14)
    Result = np.array([], dtype=int)
    print(4, 17)
    for i in range(rowNum):
        print(4, 17)
        print(5, 18)
        flag = 1
        print(7, 19)
        for j in range(columnNum):
            print(7, 19)
            if matrix1[i][j] != 0:
                print(8, 20)
                print(10, 21)
                Value = np.append(Value, np.array(matrix1[i][j]))
                print(10, 22)
                Column = np.append(Column, j)
                if flag == 1:
                    print(10, 23)
                    print(12, 24)
                    RowPtr = np.append(RowPtr, len(Value) - 1)
                    print(12, 25)
                    flag = 0
                else:
                    print(10, 23)
            else:
                print(8, 20)
    print(6, 26)
    RowPtr = np.append(RowPtr, 8)
    print(14, 30)
    for i in range(rowNum):
        print(14, 30)
        print(15, 31)
        start = int(RowPtr[i])
        print(15, 32)
        end = int(RowPtr[i + 1])
        print(15, 33)
        temp = 0
        print(17, 34)
        for j in range(start, end):
            print(17, 34)
            print(18, 35)
            k = int(Column[j])
            print(18, 36)
            temp += Value[j] * matrix2[k][0]
        print(19, 37)
        Result = np.append(Result, temp)
    return Result


if __name__ == '__main__':
    print(1, 41)
    print(22, 42)
    Result = csr(A, B)
    print('CSR result is:', Result)
else:
    print(1, 41)
