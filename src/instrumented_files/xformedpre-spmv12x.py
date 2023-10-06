import sys
from instrument_lib import *
import numpy as np
from loop import loop
print(1, 4)
A_0 = np.random.randint(0, 512, size=(400, 400))
print(1, 5)
B_0 = np.random.randint(512, high=None, size=(400, 400))


def csr(matrix1, matrix2):
    print('enter scope 1')
    print(1, 8)
    matrix1_1 = matrix1
    matrix2_1 = matrix2
    print(3, 9)
    rowNum_1 = int(matrix1_1.shape[0])
    print(3, 10)
    columnNum_1 = int(matrix1_1.shape[1])
    print(3, 11)
    Value_1 = np.array([], dtype=int)
    print(3, 12)
    Column_1 = np.array([], dtype=int)
    print(3, 13)
    RowPtr_1 = np.array([], dtype=int)
    print(3, 14)
    Result_1 = np.array([], dtype=int)
    for i_1 in range(rowNum_1):
        print(5, 18)
        flag_1 = 1
        for j_1 in range(columnNum_1):
            if matrix1_1[i_1][j_1] != 0:
                print(10, 21)
                Value_1 = np.append(Value_1, np.array(matrix1_1[i_1][j_1]))
                print(10, 22)
                Column_1 = np.append(Column_1, j_1)
                if flag_1 == 1:
                    print(12, 24)
                    RowPtr_1 = np.append(RowPtr_1, len(Value_1) - 1)
                    print(12, 25)
                    flag_1 = 0
    print(6, 26)
    RowPtr_1 = np.append(RowPtr_1, 8)
    for i_1 in range(rowNum_1):
        print(15, 31)
        start_1 = int(RowPtr_1[i_1])
        print(15, 32)
        end_1 = int(RowPtr_1[i_1 + 1])
        print(15, 33)
        temp_1 = 0
        for j_1 in range(start_1, end_1):
            print(18, 35)
            k_1 = int(Column_1[j_1])
            print(18, 36)
            temp_1 += Value_1[j_1] * matrix2_1[k_1][0]
        print(19, 37)
        Result_1 = np.append(Result_1, temp_1)
    print('exit scope 1')
    return Result_1
    print('exit scope 1')


if __name__ == '__main__':
    for i_0 in range(12):
        loop().pattern_seek()
        print(24, 44)
        Result_0 = csr(A_0, B_0)
