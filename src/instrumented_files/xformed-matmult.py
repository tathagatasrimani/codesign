import sys
from instrument_lib import *
import sys
from instrument_lib import *
from loop import loop
import numpy as np


def main():
    print('enter scope 1')
    print(1, 5)
    print(3, 6)
    a_1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
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
    b_1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
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
    c_1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
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
    for i_1 in range(3):
        for j_1 in range(3):
            instrument_read(loop, 'loop').start_unroll
            for k_1 in range(3):
                print(10, 13)
                c_1[instrument_read(i_1, 'i_1')][instrument_read(j_1, 'j_1')
                    ] += instrument_read_sub(instrument_read_sub(
                    instrument_read(a_1, 'a_1'), 'a_1', instrument_read(i_1,
                    'i_1'), None, None, False), 'a_1[i_1]', instrument_read
                    (k_1, 'k_1'), None, None, False) * instrument_read_sub(
                    instrument_read_sub(instrument_read(b_1, 'b_1'), 'b_1',
                    instrument_read(k_1, 'k_1'), None, None, False),
                    'b_1[k_1]', instrument_read(j_1, 'j_1'), None, None, False)
                write_instrument_read_sub(c_1[instrument_read(
                    instrument_read(i_1, 'i_1'), 'i_1')],
                    "c_1[instrument_read(i_1, 'i_1')]", instrument_read(
                    instrument_read(j_1, 'j_1'), 'j_1'), None, None, False)
            instrument_read(loop, 'loop').stop_unroll
    print('exit scope 1')


if instrument_read(__name__, '__name__') == '__main__':
    main()
