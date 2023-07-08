digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="if __name__ == '__main__':
"]
	2 [label="matrix = [[0]]
"]
	4 [label="for i in range(1):
"]
	5 [label="for j in range(1):
"]
	7 [label="a = matrix[i][j]
"]
	7 -> 5 [label=""]
	5 -> 7 [label="range(1)"]
	5 -> 4 [label=""]
	4 -> 5 [label="range(1)"]
	2 -> 4 [label=""]
	1 -> 2 [label="__name__ == '__main__'"]
}
