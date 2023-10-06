digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="from loop import loop
import numpy as np
def mult(a, b):...
if __name__ == '__main__':
"]
	9 [label="mult(2, 3)
"]
	"9_calls" [label=mult shape=box]
	9 -> "9_calls" [label=calls style=dashed]
	1 -> 9 [label="__name__ == '__main__'"]
	subgraph clustermult {
		graph [label=mult]
		3 [label="a_1 = a
b_1 = b
res_1 = 0
"]
		4 [label="for i_1 in range(a_1):
"]
		5 [label="res_1 += b_1
"]
		5 -> 4 [label=""]
		4 -> 5 [label="range(a_1)"]
		6 [label="return res_1
"]
		4 -> 6 [label=""]
		3 -> 4 [label=""]
	}
}
