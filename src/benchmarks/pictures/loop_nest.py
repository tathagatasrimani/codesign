digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import loop as loop
import numpy as np
def main():...
if __name__ == '__main__':
"]
	11 [label="main()
"]
	"11_calls" [label=main shape=box]
	11 -> "11_calls" [label=calls style=dashed]
	1 -> 11 [label="__name__ == '__main__'"]
	subgraph clustermain {
		graph [label=main]
		3 [label="a_1 = 3
b_1 = 2
c_1 = 0
d_1 = 0
"]
		4 [label="for i_1 in range(10):
"]
		5 [label="for j_1 in range(10):
"]
		7 [label="c_1 += a_1 * b_1
d_1 *= a_1 + b_1
"]
		7 -> 5 [label=""]
		5 -> 7 [label="range(10)"]
		5 -> 4 [label=""]
		4 -> 5 [label="range(10)"]
		6 [label="return c_1
"]
		4 -> 6 [label=""]
		3 -> 4 [label=""]
	}
}
