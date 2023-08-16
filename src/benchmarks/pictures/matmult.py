digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="from loop import loop
import numpy as np
def main():...
if __name__ == '__main__':
"]
	13 [label="main()
"]
	"13_calls" [label=main shape=box]
	13 -> "13_calls" [label=calls style=dashed]
	1 -> 13 [label="__name__ == '__main__'"]
	subgraph clustermain {
		graph [label=main]
		3 [label="a_1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
b_1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
c_1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
"]
		4 [label="for i_1 in range(3):
"]
		5 [label="for j_1 in range(3):
"]
		7 [label="loop.start_unroll
"]
		9 [label="for k_1 in range(3):
"]
		10 [label="c_1[i_1][j_1] += a_1[i_1][k_1] * b_1[k_1][j_1]
"]
		10 -> 9 [label=""]
		9 -> 10 [label="range(3)"]
		11 [label="loop.stop_unroll
"]
		11 -> 5 [label=""]
		9 -> 11 [label=""]
		7 -> 9 [label=""]
		5 -> 7 [label="range(3)"]
		5 -> 4 [label=""]
		4 -> 5 [label="range(3)"]
		3 -> 4 [label=""]
	}
}
