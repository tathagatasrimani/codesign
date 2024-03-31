digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import math
import cmath as cm
import numpy as np
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
		3 [label="x_1 = x
\"\"\"
    Fast Fourier Transform using a recursive decimation in time algorithm. This has O(N log_2(N))
    complexity. This implementation uses native Python lists.

    :Parameters:
      x
        The vector of which the FFT will be computed. This should always be called with a vector of
        a power of two length, or it will fail. No checks on this are made.

    :Returns:
      A complex-number vector of the same size, with the coefficients of the DFT.
    \"\"\"
if len(x_1) == 1:
"]
		4 [label="return x_1
"]
		3 -> 4 [label="len(x_1) == 1"]
		6 [label="N_1 = len(x_1)
Xe_1 = recursive_fft(x_1[0::2])
Xo_1 = recursive_fft(x_1[1::2])
W_1 = [cm_1.exp(-2.0j * cm_1.pi * k_1 / N_1) for k_1 in range(N_1 // 2)]
WXo_1 = [(Wk_1 * Xok_1) for Wk_1, Xok_1 in zip(W_1, Xo_1)]
X_1 = [(Xek_1 + WXok_1) for Xek_1, WXok_1 in zip(Xe_1, WXo_1)] + [(Xek_1 -
    WXok_1) for Xek_1, WXok_1 in zip(Xe_1, WXo_1)]
return X_1
"]
		"6_calls" [label="len
recursive_fft
recursive_fft
cm_1.exp
range
zip
zip
zip" shape=box]
		6 -> "6_calls" [label=calls style=dashed]
		3 -> 6 [label="(len(x_1) != 1)"]
	}
}
