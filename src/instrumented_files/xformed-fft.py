import sys
from instrument_lib import *
import math
import cmath as cm
import numpy as np


def recursive_fft(x):
    print('enter scope 1')
    print(1, 6)
    print(3, 7)
    x_1 = instrument_read(x, 'x')
    write_instrument_read(x_1, 'x_1')
    if type(x_1) == np.ndarray:
        print('malloc', sys.getsizeof(x_1), 'x_1', x_1.shape)
    elif type(x_1) == list:
        dims = []
        tmp = x_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(x_1), 'x_1', dims)
    elif type(x_1) == tuple:
        print('malloc', sys.getsizeof(x_1), 'x_1', [len(x_1)])
    else:
        print('malloc', sys.getsizeof(x_1), 'x_1')
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
    if len(instrument_read(x_1, 'x_1')) == 1:
        print('exit scope 1')
        return instrument_read(x_1, 'x_1')
    else:
        print(6, 23)
        N_1 = len(instrument_read(x_1, 'x_1'))
        write_instrument_read(N_1, 'N_1')
        if type(N_1) == np.ndarray:
            print('malloc', sys.getsizeof(N_1), 'N_1', N_1.shape)
        elif type(N_1) == list:
            dims = []
            tmp = N_1
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(N_1), 'N_1', dims)
        elif type(N_1) == tuple:
            print('malloc', sys.getsizeof(N_1), 'N_1', [len(N_1)])
        else:
            print('malloc', sys.getsizeof(N_1), 'N_1')
        print(6, 24)
        Xe_1 = recursive_fft(instrument_read_sub(instrument_read(x_1, 'x_1'
            ), 'x_1', None, 0, None, True))
        write_instrument_read(Xe_1, 'Xe_1')
        if type(Xe_1) == np.ndarray:
            print('malloc', sys.getsizeof(Xe_1), 'Xe_1', Xe_1.shape)
        elif type(Xe_1) == list:
            dims = []
            tmp = Xe_1
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(Xe_1), 'Xe_1', dims)
        elif type(Xe_1) == tuple:
            print('malloc', sys.getsizeof(Xe_1), 'Xe_1', [len(Xe_1)])
        else:
            print('malloc', sys.getsizeof(Xe_1), 'Xe_1')
        print(6, 25)
        Xo_1 = recursive_fft(instrument_read_sub(instrument_read(x_1, 'x_1'
            ), 'x_1', None, 1, None, True))
        write_instrument_read(Xo_1, 'Xo_1')
        if type(Xo_1) == np.ndarray:
            print('malloc', sys.getsizeof(Xo_1), 'Xo_1', Xo_1.shape)
        elif type(Xo_1) == list:
            dims = []
            tmp = Xo_1
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(Xo_1), 'Xo_1', dims)
        elif type(Xo_1) == tuple:
            print('malloc', sys.getsizeof(Xo_1), 'Xo_1', [len(Xo_1)])
        else:
            print('malloc', sys.getsizeof(Xo_1), 'Xo_1')
        print(6, 26)
        W_1 = [instrument_read(cm_1, 'cm_1').exp(-2.0j * instrument_read(
            cm_1, 'cm_1').pi * instrument_read(k_1, 'k_1') /
            instrument_read(N_1, 'N_1')) for k_1 in range(instrument_read(
            N_1, 'N_1') // 2)]
        write_instrument_read(W_1, 'W_1')
        if type(W_1) == np.ndarray:
            print('malloc', sys.getsizeof(W_1), 'W_1', W_1.shape)
        elif type(W_1) == list:
            dims = []
            tmp = W_1
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(W_1), 'W_1', dims)
        elif type(W_1) == tuple:
            print('malloc', sys.getsizeof(W_1), 'W_1', [len(W_1)])
        else:
            print('malloc', sys.getsizeof(W_1), 'W_1')
        print(6, 28)
        WXo_1 = [(instrument_read(Wk_1, 'Wk_1') * instrument_read(Xok_1,
            'Xok_1')) for Wk_1, Xok_1 in zip(instrument_read(W_1, 'W_1'),
            instrument_read(Xo_1, 'Xo_1'))]
        write_instrument_read(WXo_1, 'WXo_1')
        if type(WXo_1) == np.ndarray:
            print('malloc', sys.getsizeof(WXo_1), 'WXo_1', WXo_1.shape)
        elif type(WXo_1) == list:
            dims = []
            tmp = WXo_1
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(WXo_1), 'WXo_1', dims)
        elif type(WXo_1) == tuple:
            print('malloc', sys.getsizeof(WXo_1), 'WXo_1', [len(WXo_1)])
        else:
            print('malloc', sys.getsizeof(WXo_1), 'WXo_1')
        print(6, 29)
        X_1 = [(instrument_read(Xek_1, 'Xek_1') + instrument_read(WXok_1,
            'WXok_1')) for Xek_1, WXok_1 in zip(instrument_read(Xe_1,
            'Xe_1'), instrument_read(WXo_1, 'WXo_1'))] + [(instrument_read(
            Xek_1, 'Xek_1') - instrument_read(WXok_1, 'WXok_1')) for Xek_1,
            WXok_1 in zip(instrument_read(Xe_1, 'Xe_1'), instrument_read(
            WXo_1, 'WXo_1'))]
        write_instrument_read(X_1, 'X_1')
        if type(X_1) == np.ndarray:
            print('malloc', sys.getsizeof(X_1), 'X_1', X_1.shape)
        elif type(X_1) == list:
            dims = []
            tmp = X_1
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(X_1), 'X_1', dims)
        elif type(X_1) == tuple:
            print('malloc', sys.getsizeof(X_1), 'X_1', [len(X_1)])
        else:
            print('malloc', sys.getsizeof(X_1), 'X_1')
        print('exit scope 1')
        return instrument_read(X_1, 'X_1')
    print('exit scope 1')


if instrument_read(__name__, '__name__') == '__main__':
    print(recursive_fft([1, 2, 3, 4]))
