digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="def bruh():...
def main():...
if __name__ == '__main__':
"]
	11 [label="main()
"]
	"11_calls" [label=main shape=box]
	11 -> "11_calls" [label=calls style=dashed]
	1 -> 11 [label="__name__ == '__main__'"]
	subgraph clusterbruh {
		graph [label=bruh]
		3 [label="return 3, 4
"]
	}
	subgraph clustermain {
		graph [label=main]
		7 [label="i, j = 1, 2
b = [1, 2, 3]
k, l = bruh()
c: int
d: int = 5
b[0] = 2
if 3 < 4 < 5 > 6:
"]
		"7_calls" [label=bruh shape=box]
		7 -> "7_calls" [label=calls style=dashed]
		8 [label="c = 2
"]
		7 -> 8 [label="3 < 4 < 5 > 6"]
	}
}
