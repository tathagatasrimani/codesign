import sys
from memory import Memory
MEMORY_SIZE = 10000
import math
import cmath as cm


def recursive_fft(x):
    print(1, 5)
    memory_module = Memory(MEMORY_SIZE)
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
    if len(x) == 1:
        print(3, 18)
        return x
    else:
        print(3, 18)
        print(6, 21)
        N = len(x)
        memory_module.malloc('N', sys.getsizeof(N))
        print(memory_module.locations['N'].location, 'N', 'mem')
        print(6, 22)
        Xe = recursive_fft(x[0::2])
        memory_module.malloc('Xe', sys.getsizeof(Xe))
        print(memory_module.locations['Xe'].location, 'Xe', 'mem')
        print(6, 23)
        Xo = recursive_fft(x[1::2])
        memory_module.malloc('Xo', sys.getsizeof(Xo))
        print(memory_module.locations['Xo'].location, 'Xo', 'mem')
        print(6, 24)
        W = [cm.exp(-2.0j * cm.pi * k / N) for k in range(N // 2)]
        memory_module.malloc('W', sys.getsizeof(W))
        print(memory_module.locations['W'].location, 'W', 'mem')
        print(6, 25)
        WXo = [(Wk * Xok) for Wk, Xok in zip(W, Xo)]
        memory_module.malloc('WXo', sys.getsizeof(WXo))
        print(memory_module.locations['WXo'].location, 'WXo', 'mem')
        print(6, 26)
        X = [(Xek + WXok) for Xek, WXok in zip(Xe, WXo)] + [(Xek - WXok) for
            Xek, WXok in zip(Xe, WXo)]
        memory_module.malloc('X', sys.getsizeof(X))
        print(memory_module.locations['X'].location, 'X', 'mem')
        return X


if __name__ == '__main__':
    print(1, 32)
    print(recursive_fft([1, 2, 3, 4]))
else:
    print(1, 32)
