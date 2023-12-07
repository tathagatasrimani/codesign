import math  # Mathematical operations;
import cmath as cm  # Complex math;


def recursive_fft(x):
    """
    Fast Fourier Transform using a recursive decimation in time algorithm. This has O(N log_2(N))
    complexity. This implementation uses native Python lists.

    :Parameters:
      x
        The vector of which the FFT will be computed. This should always be called with a vector of
        a power of two length, or it will fail. No checks on this are made.

    :Returns:
      A complex-number vector of the same size, with the coefficients of the DFT.
    """
    if len(x) == 1:  # A length-1 vector is its own FT;
        return x
    else:
        N = len(x)  # Length of the vector;
        Xe = recursive_fft(x[0::2])  # Transform of even samples;
        Xo = recursive_fft(x[1::2])  # Transform of odd samples;
        W = [cm.exp(-2j * cm.pi * k / N) for k in range(N // 2)]  # Twiddle factors;
        WXo = [Wk * Xok for Wk, Xok in zip(W, Xo)]
        X = [Xek + WXok for Xek, WXok in zip(Xe, WXo)] + [  # Recombine results;
            Xek - WXok for Xek, WXok in zip(Xe, WXo)
        ]
        return X


if __name__ == "__main__":
    print(recursive_fft([1, 2, 3, 4]))
