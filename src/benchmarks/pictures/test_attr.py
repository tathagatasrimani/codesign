digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="def main():...
if __name__ == '__main__':
"]
	5 [label="main()
"]
	"5_calls" [label=main shape=box]
	5 -> "5_calls" [label=calls style=dashed]
	1 -> 5 [label="__name__ == '__main__'"]
	subgraph clustermain {
		graph [label=main]
		3 [label="b_5 = 2
b_6 = 3
e = b_5
"]
	}
}
