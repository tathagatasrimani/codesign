digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="if __name__ == '__main__':
"]
	2 [label="a = 2
b = a
"]
	1 -> 2 [label="__name__ == '__main__'"]
}
