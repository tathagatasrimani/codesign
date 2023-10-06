import numpy as np
from loop import loop
#CSR format
A = np.random.randint(0, 512, size=(400,400))
B = np.random.randint(512, high=None, size=(400, 400))


def csr(matrix1, matrix2):
    rowNum = int(matrix1.shape[0])
    columnNum = int(matrix1.shape[1])
    Value = np.array([], dtype = int)
    Column = np.array([], dtype = int)
    RowPtr = np.array([], dtype = int)
    Result = np.array([], dtype = int)


    for i in range(rowNum):
        flag = 1
        for j in range(columnNum):
            if matrix1[i][j] != 0:
               Value = np.append(Value, np.array(matrix1[i][j]))
               Column = np.append(Column, j)
               if flag == 1:
                   RowPtr = np.append(RowPtr, len(Value)-1)
                   flag = 0
    RowPtr = np.append(RowPtr, 8)


#CSR kernel: A x B
    for i in range(rowNum):
        start = int(RowPtr[i])
        end = int(RowPtr[i + 1])
        temp = 0
        for j in range(start, end):
            k = int(Column[j])
            temp += Value[j] * matrix2[k][0]
        Result = np.append(Result, temp)
    return Result


if __name__ == "__main__":
    Result = csr(A,B)
    print ("CSR result is:", Result)
