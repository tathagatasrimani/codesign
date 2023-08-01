digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="from loop import loop
def main(x, y):...
def bruh():...
if __name__ == '__main__':
"]
	19 [label="main(2, 3)
bruh()
"]
	"19_calls" [label="main
bruh" shape=box]
	19 -> "19_calls" [label=calls style=dashed]
	1 -> 19 [label="__name__ == '__main__'"]
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
		5 [label="z = [[1, 2, 3, 4, 5, 6]]
r, q = 2, 3
loop.start_unroll
"]
		7 [label="for i in range(5):
"]
		8 [label="loop().pattern_seek()
z[0][i] += z[0][i + 1]
if r > 0:
"]
		"8_calls" [label="Call.pattern_seek" shape=box]
		8 -> "8_calls" [label=calls style=dashed]
		10 [label="q += 3
"]
		10 -> 7 [label=""]
		8 -> 10 [label="r > 0"]
		8 -> 7 [label="(r <= 0)"]
		7 -> 8 [label="range(5)"]
		9 [label="loop.stop_unroll
"]
		7 -> 9 [label=""]
		5 -> 7 [label=""]
		4 -> 5 [label=""]
		3 -> 4 [label="w < 0"]
		6 [label="a = q - 3
b = a / r
"]
		6 -> 5 [label=""]
		3 -> 6 [label="(w >= 0)"]
	}
	subgraph clusterbruh {
		graph [label=bruh]
		14 [label="a = 1
loop.start_unroll
"]
		15 [label="for i in range(3):
"]
		16 [label="a += i
"]
		16 -> 15 [label=""]
		15 -> 16 [label="range(3)"]
		17 [label="loop.stop_unroll
"]
		15 -> 17 [label=""]
		14 -> 15 [label=""]
	}
}
