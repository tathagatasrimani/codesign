import sys
from instrument_lib import *
import sys
from instrument_lib import *
from loop import loop
import numpy as np


def mult(a, b):
    print('enter scope 1')
    print(1, 5)
    print(3, 6)
    a_1 = instrument_read(a, 'a')
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
    b_1 = instrument_read(b, 'b')
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
    res_1 = 0
    write_instrument_read(res_1, 'res_1')
    if type(res_1) == np.ndarray:
        print('malloc', sys.getsizeof(res_1), 'res_1', res_1.shape)
    elif type(res_1) == list:
        dims = []
        tmp = res_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(res_1), 'res_1', dims)
    elif type(res_1) == tuple:
        print('malloc', sys.getsizeof(res_1), 'res_1', [len(res_1)])
    else:
        print('malloc', sys.getsizeof(res_1), 'res_1')
    for i_1 in range(instrument_read(a_1, 'a_1')):
        print(5, 10)
        res_1 += instrument_read(b_1, 'b_1')
        write_instrument_read(res_1, 'res_1')
    print('exit scope 1')
    return instrument_read(res_1, 'res_1')
    print('exit scope 1')


if instrument_read(__name__, '__name__') == '__main__':
    mult(2, 3)
