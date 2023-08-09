import sys
from instrument_lib import *
import sys
from instrument_lib import *
from loop import loop
import numpy as np


def main(x, y):
    print('enter scope 1')
    print(1, 4)
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
    else:
        print('malloc', sys.getsizeof(x_1), 'x_1')
    y_1 = instrument_read(y, 'y')
    write_instrument_read(y_1, 'y_1')
    if type(y_1) == np.ndarray:
        print('malloc', sys.getsizeof(y_1), 'y_1', y_1.shape)
    elif type(y_1) == list:
        dims = []
        tmp = y_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(y_1), 'y_1', dims)
    else:
        print('malloc', sys.getsizeof(y_1), 'y_1')
    print(3, 5)
    q_1 = 0.5 + instrument_read(x_1, 'x_1') * instrument_read(y_1, 'y_1'
        ) + 1 / 2
    write_instrument_read(q_1, 'q_1')
    if type(q_1) == np.ndarray:
        print('malloc', sys.getsizeof(q_1), 'q_1', q_1.shape)
    elif type(q_1) == list:
        dims = []
        tmp = q_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(q_1), 'q_1', dims)
    else:
        print('malloc', sys.getsizeof(q_1), 'q_1')
    print(3, 6)
    r_1 = instrument_read(x_1, 'x_1') + instrument_read(y_1, 'y_1')
    write_instrument_read(r_1, 'r_1')
    if type(r_1) == np.ndarray:
        print('malloc', sys.getsizeof(r_1), 'r_1', r_1.shape)
    elif type(r_1) == list:
        dims = []
        tmp = r_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(r_1), 'r_1', dims)
    else:
        print('malloc', sys.getsizeof(r_1), 'r_1')
    print(3, 7)
    q_1 = instrument_read(q_1, 'q_1') * instrument_read(r_1, 'r_1') - 3
    write_instrument_read(q_1, 'q_1')
    if type(q_1) == np.ndarray:
        print('malloc', sys.getsizeof(q_1), 'q_1', q_1.shape)
    elif type(q_1) == list:
        dims = []
        tmp = q_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(q_1), 'q_1', dims)
    else:
        print('malloc', sys.getsizeof(q_1), 'q_1')
    print(3, 8)
    w_1 = instrument_read(q_1, 'q_1') + instrument_read(r_1, 'r_1')
    write_instrument_read(w_1, 'w_1')
    if type(w_1) == np.ndarray:
        print('malloc', sys.getsizeof(w_1), 'w_1', w_1.shape)
    elif type(w_1) == list:
        dims = []
        tmp = w_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(w_1), 'w_1', dims)
    else:
        print('malloc', sys.getsizeof(w_1), 'w_1')
    if instrument_read(w_1, 'w_1') < 0:
        print(4, 10)
        a_1 = instrument_read(q_1, 'q_1') + 3
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
        else:
            print('malloc', sys.getsizeof(a_1), 'a_1')
        print(4, 11)
        b_1 = instrument_read(a_1, 'a_1') * instrument_read(r_1, 'r_1')
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
        else:
            print('malloc', sys.getsizeof(b_1), 'b_1')
        print(4, 12)
        r_1 += instrument_read(a_1, 'a_1') + 3 * 2
        write_instrument_read(r_1, 'r_1')
    else:
        print(6, 14)
        a_1 = instrument_read(q_1, 'q_1') - 3
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
        else:
            print('malloc', sys.getsizeof(a_1), 'a_1')
        print(6, 15)
        b_1 = instrument_read(a_1, 'a_1') / instrument_read(r_1, 'r_1')
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
        else:
            print('malloc', sys.getsizeof(b_1), 'b_1')
    print(5, 16)
    z_1 = [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z_1, 'z_1')
    if type(z_1) == np.ndarray:
        print('malloc', sys.getsizeof(z_1), 'z_1', z_1.shape)
    elif type(z_1) == list:
        dims = []
        tmp = z_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(z_1), 'z_1', dims)
    else:
        print('malloc', sys.getsizeof(z_1), 'z_1')
    print(5, 17)
    d_1 = [[[1], [1]], [[1], [1]]]
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
    else:
        print('malloc', sys.getsizeof(d_1), 'd_1')
    print(5, 18)
    r_1, q_1 = 2, 3
    write_instrument_read(q_1, 'q_1')
    if type(q_1) == np.ndarray:
        print('malloc', sys.getsizeof(q_1), 'q_1', q_1.shape)
    elif type(q_1) == list:
        dims = []
        tmp = q_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(q_1), 'q_1', dims)
    else:
        print('malloc', sys.getsizeof(q_1), 'q_1')
    print(5, 19)
    g_1 = instrument_read(np, 'np').zeros((5, 4, 2))
    write_instrument_read(g_1, 'g_1')
    if type(g_1) == np.ndarray:
        print('malloc', sys.getsizeof(g_1), 'g_1', g_1.shape)
    elif type(g_1) == list:
        dims = []
        tmp = g_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(g_1), 'g_1', dims)
    else:
        print('malloc', sys.getsizeof(g_1), 'g_1')
    instrument_read(loop, 'loop').start_unroll
    for i_1 in range(5):
        print(8, 22)
        z_1[0][1] = 1
        write_instrument_read_sub(z_1[0], 'z_1[0]', 1, None, None, False)
        print(8, 23)
        z_1[0][instrument_read(i_1, 'i_1')] += instrument_read_sub(
            instrument_read_sub(instrument_read(z_1, 'z_1'), 'z_1', 0, None,
            None, False), 'z_1[0]', instrument_read(i_1, 'i_1') + 1, None,
            None, False)
        write_instrument_read_sub(z_1[0], 'z_1[0]', instrument_read(
            instrument_read(i_1, 'i_1'), 'i_1'), None, None, False)
    instrument_read(loop, 'loop').stop_unroll
    print('exit scope 1')


def bruh():
    print('enter scope 2')
    print(1, 26)
    print(12, 27)
    a_2 = 1
    write_instrument_read(a_2, 'a_2')
    if type(a_2) == np.ndarray:
        print('malloc', sys.getsizeof(a_2), 'a_2', a_2.shape)
    elif type(a_2) == list:
        dims = []
        tmp = a_2
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(a_2), 'a_2', dims)
    else:
        print('malloc', sys.getsizeof(a_2), 'a_2')
    instrument_read(loop, 'loop').start_unroll
    for i_2 in range(3):
        print(14, 30)
        a_2 += instrument_read(i_2, 'i_2')
        write_instrument_read(a_2, 'a_2')
    instrument_read(loop, 'loop').stop_unroll
    print('exit scope 2')


if instrument_read(__name__, '__name__') == '__main__':
    for i_0 in range(1):
        loop().pattern_seek()
        main(2, 3)
        bruh()
