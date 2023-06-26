digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="def main(x, y):...
if __name__ == '__main__':
"]
	11 [label="main(2, 3)
"]
	"11_calls" [label=main shape=box]
	11 -> "11_calls" [label=calls style=dashed]
	1 -> 11 [label="__name__ == '__main__'"]
	subgraph clustermain {
		graph [label=main]
		3 [label="q = 0.5 + x * y + 1 / 2
r = x + y
q = q * r - 3
w = q + r
if w < 0:
"]
		4 [label="a = q + 3
b = a * r
r += a + 3 * 2
"]
		5 [label="z = [1, 2, 3, 4, 5]
z += [1, 2, 3, 4, 5]
"]
		7 [label="for i in range(5):
"]
		8 [label="z[i] += 1
"]
		8 -> 7 [label=""]
		7 -> 8 [label="range(5)"]
		5 -> 7 [label=""]
		4 -> 5 [label=""]
		3 -> 4 [label="w < 0"]
		6 [label="a = q - 3
b = a / r
"]
		6 -> 5 [label=""]
		3 -> 6 [label="(w >= 0)"]
	}
}
