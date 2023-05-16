digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="if __name__ == '__main__':
"]
	2 [label="matrix = []
bruh = 2
matrix.append(2)
len(matrix[bruh - 1:][0])
print(matrix)
"]
	"2_calls" [label="matrix.append
len
print" shape=box]
	2 -> "2_calls" [label=calls style=dashed]
	1 -> 2 [label="__name__ == '__main__'"]
}
