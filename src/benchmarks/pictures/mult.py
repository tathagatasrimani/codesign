digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import numpy as np
def main():...
if __name__ == '__main__':
"]
	5 [label="main()
"]
	"5_calls" [label=main shape=box]
	5 -> "5_calls" [label=calls style=dashed]
	1 -> 5 [label="__name__ == '__main__'"]
	subgraph clustermain {
		graph [label=main]
		3 [label="a_1 = 2
b_1 = 3
c_1 = a_1 * b_1
"]
	}
}
