import sys
from instrument_lib import *
import sys
from instrument_lib import *
import loop as loop
import numpy as np


def main():
    print('enter scope 1')
    print(1, 5)
    print(3, 6)
    a_1 = 3
    write_instrument_read(a_1, 'a_1')
    if type(a_1) == np.ndarray:
        print('malloc', sys.getsizeof(a_1), 'a_1', a_1.shape)
    elif type(a_1) == list:
        dims = []
        tmp = a_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(a_1), 'a_1', dims)
    elif type(a_1) == tuple:
        print('malloc', sys.getsizeof(a_1), 'a_1', [len(a_1)])
    else:
        print('malloc', sys.getsizeof(a_1), 'a_1')
    print(3, 7)
    b_1 = 2
    write_instrument_read(b_1, 'b_1')
    if type(b_1) == np.ndarray:
        print('malloc', sys.getsizeof(b_1), 'b_1', b_1.shape)
    elif type(b_1) == list:
        dims = []
        tmp = b_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(b_1), 'b_1', dims)
    elif type(b_1) == tuple:
        print('malloc', sys.getsizeof(b_1), 'b_1', [len(b_1)])
    else:
        print('malloc', sys.getsizeof(b_1), 'b_1')
    print(3, 8)
    c_1 = 0
    write_instrument_read(c_1, 'c_1')
    if type(c_1) == np.ndarray:
        print('malloc', sys.getsizeof(c_1), 'c_1', c_1.shape)
    elif type(c_1) == list:
        dims = []
        tmp = c_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(c_1), 'c_1', dims)
    elif type(c_1) == tuple:
        print('malloc', sys.getsizeof(c_1), 'c_1', [len(c_1)])
    else:
        print('malloc', sys.getsizeof(c_1), 'c_1')
    print(3, 9)
    d_1 = 0
    write_instrument_read(d_1, 'd_1')
    if type(d_1) == np.ndarray:
        print('malloc', sys.getsizeof(d_1), 'd_1', d_1.shape)
    elif type(d_1) == list:
        dims = []
        tmp = d_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(d_1), 'd_1', dims)
    elif type(d_1) == tuple:
        print('malloc', sys.getsizeof(d_1), 'd_1', [len(d_1)])
    else:
        print('malloc', sys.getsizeof(d_1), 'd_1')
    for i_1 in range(10):
        for j_1 in range(10):
            print(7, 12)
            c_1 += instrument_read(a_1, 'a_1') * instrument_read(b_1, 'b_1')
            write_instrument_read(c_1, 'c_1')
            print(7, 13)
            d_1 *= instrument_read(a_1, 'a_1') + instrument_read(b_1, 'b_1')
            write_instrument_read(d_1, 'd_1')
    print('exit scope 1')
    return instrument_read(c_1, 'c_1')
    print('exit scope 1')


if instrument_read(__name__, '__name__') == '__main__':
    main()
