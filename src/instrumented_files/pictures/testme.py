digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="from loop import loop
import numpy as np
def main(x, y):...
def bruh():...
if __name__ == '__main__':
"]
	17 [label="for i_0 in range(2):
"]
	19 [label="loop().pattern_seek()
main(2, 3)
bruh()
"]
	"19_calls" [label="Call.pattern_seek
main
bruh" shape=box]
	19 -> "19_calls" [label=calls style=dashed]
	19 -> 17 [label=""]
	17 -> 19 [label="range(2)"]
	1 -> 17 [label="__name__ == '__main__'"]
	subgraph clustermain {
		graph [label=main]
		3 [label="x_1 = x
y_1 = y
q_1 = 0.5 + x_1 * y_1 + 1 / 2
r_1 = x_1 + y_1
q_1 = q_1 * r_1 - 3
w_1 = q_1 + r_1
if w_1 < 0:
"]
		4 [label="a_1 = q_1 + 3
b_1 = a_1 * r_1
r_1 += a_1 + 3 * 2
"]
		5 [label="z_1 = [[1, 2, 3, 4, 5, 6]]
r_1, q_1 = 2, 3
loop.start_unroll
"]
		7 [label="for i_1 in range(5):
"]
		8 [label="z_1[0][i_1] += z_1[0][i_1 + 1]
"]
		8 -> 7 [label=""]
		7 -> 8 [label="range(5)"]
		9 [label="loop.stop_unroll
"]
		7 -> 9 [label=""]
		5 -> 7 [label=""]
		4 -> 5 [label=""]
		3 -> 4 [label="w_1 < 0"]
		6 [label="a_1 = q_1 - 3
b_1 = a_1 / r_1
"]
		6 -> 5 [label=""]
		3 -> 6 [label="(w_1 >= 0)"]
	}
	subgraph clusterbruh {
		graph [label=bruh]
		12 [label="a_2 = 1
loop.start_unroll
"]
		13 [label="for i_2 in range(3):
"]
		14 [label="a_2 += i_2
"]
		14 -> 13 [label=""]
		13 -> 14 [label="range(3)"]
		15 [label="loop.stop_unroll
"]
		13 -> 15 [label=""]
		12 -> 13 [label=""]
	}
}
