digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import math
import cmath as cm
def recursive_fft(x):...
if __name__ == '__main__':
"]
	10 [label="print(recursive_fft([1, 2, 3, 4]))
"]
	"10_calls" [label=print shape=box]
	10 -> "10_calls" [label=calls style=dashed]
	1 -> 10 [label="__name__ == '__main__'"]
	subgraph clusterrecursive_fft {
		graph [label=recursive_fft]
		3 [label="\"\"\"
    Fast Fourier Transform using a recursive decimation in time algorithm. This has O(N log_2(N))
    complexity. This implementation uses native Python lists.

    :Parameters:
      x
        The vector of which the FFT will be computed. This should always be called with a vector of
        a power of two length, or it will fail. No checks on this are made.

    :Returns:
      A complex-number vector of the same size, with the coefficients of the DFT.
    \"\"\"
if len(x) == 1:
"]
		4 [label="return x
"]
		3 -> 4 [label="len(x) == 1"]
		6 [label="N = len(x)
Xe = recursive_fft(x[0::2])
Xo = recursive_fft(x[1::2])
W = [cm.exp(-2.0j * cm.pi * k / N) for k in range(N // 2)]
WXo = [(Wk * Xok) for Wk, Xok in zip(W, Xo)]
X = [(Xek + WXok) for Xek, WXok in zip(Xe, WXo)] + [(Xek - WXok) for Xek,
    WXok in zip(Xe, WXo)]
return X
"]
		"6_calls" [label="len
recursive_fft
recursive_fft
cm.exp
range
zip
zip
zip" shape=box]
		6 -> "6_calls" [label=calls style=dashed]
		3 -> 6 [label="(len(x) != 1)"]
	}
}
