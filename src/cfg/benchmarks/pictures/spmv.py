digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import numpy as np
A = np.random.randint(0, 512, size=(10, 10))
B = np.random.randint(512, high=None, size=(10, 10))
def csr(matrix1, matrix2):...
if __name__ == '__main__':
"]
	"1_calls" [label="np.random.randint
np.random.randint" shape=box]
	1 -> "1_calls" [label=calls style=dashed]
	22 [label="Result = csr(A, B)
print('CSR result is:', Result)
"]
	"22_calls" [label="csr
print" shape=box]
	22 -> "22_calls" [label=calls style=dashed]
	1 -> 22 [label="__name__ == '__main__'"]
	subgraph clustercsr {
		graph [label=csr]
		3 [label="rowNum = int(matrix1.shape[0])
columnNum = int(matrix1.shape[1])
Value = np.array([], dtype=int)
Column = np.array([], dtype=int)
RowPtr = np.array([], dtype=int)
Result = np.array([], dtype=int)
"]
		"3_calls" [label="int
int
np.array
np.array
np.array
np.array" shape=box]
		3 -> "3_calls" [label=calls style=dashed]
		4 [label="for i in range(rowNum):
"]
		5 [label="flag = 1
"]
		7 [label="for j in range(columnNum):
"]
		8 [label="if matrix1[i][j] != 0:
"]
		10 [label="Value = np.append(Value, np.array(matrix1[i][j]))
Column = np.append(Column, j)
if flag == 1:
"]
		"10_calls" [label="np.append
np.append" shape=box]
		10 -> "10_calls" [label=calls style=dashed]
		12 [label="RowPtr = np.append(RowPtr, len(Value) - 1)
flag = 0
"]
		"12_calls" [label="np.append" shape=box]
		12 -> "12_calls" [label=calls style=dashed]
		12 -> 7 [label=""]
		10 -> 12 [label="flag == 1"]
		10 -> 7 [label="(flag != 1)"]
		8 -> 10 [label="matrix1[i][j] != 0"]
		8 -> 7 [label="(matrix1[i][j] == 0)"]
		7 -> 8 [label="range(columnNum)"]
		7 -> 4 [label=""]
		5 -> 7 [label=""]
		4 -> 5 [label="range(rowNum)"]
		6 [label="RowPtr = np.append(RowPtr, 8)
"]
		"6_calls" [label="np.append" shape=box]
		6 -> "6_calls" [label=calls style=dashed]
		14 [label="for i in range(rowNum):
"]
		15 [label="start = int(RowPtr[i])
end = int(RowPtr[i + 1])
temp = 0
"]
		"15_calls" [label="int
int" shape=box]
		15 -> "15_calls" [label=calls style=dashed]
		17 [label="for j in range(start, end):
"]
		18 [label="k = int(Column[j])
temp += Value[j] * matrix2[k][0]
"]
		"18_calls" [label=int shape=box]
		18 -> "18_calls" [label=calls style=dashed]
		18 -> 17 [label=""]
		17 -> 18 [label="range(start, end)"]
		19 [label="Result = np.append(Result, temp)
"]
		"19_calls" [label="np.append" shape=box]
		19 -> "19_calls" [label=calls style=dashed]
		19 -> 14 [label=""]
		17 -> 19 [label=""]
		15 -> 17 [label=""]
		14 -> 15 [label="range(rowNum)"]
		16 [label="return Result
"]
		14 -> 16 [label=""]
		6 -> 14 [label=""]
		4 -> 6 [label=""]
		3 -> 4 [label=""]
	}
}
