digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="def main(a, b):...
if __name__ == '__main__':
"]
	6 [label="a = 1
main(a, b=a)
"]
	"6_calls" [label=main shape=box]
	6 -> "6_calls" [label=calls style=dashed]
	1 -> 6 [label="__name__ == '__main__'"]
	subgraph clustermain {
		graph [label=main]
		3 [label="return a
"]
	}
}
