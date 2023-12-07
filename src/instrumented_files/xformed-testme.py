import sys
from instrument_lib import *
from loop import loop
import numpy as np


def main(x, y):
    print('enter scope 1')
    print(1, 5)
    print(3, 6)
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
    print(3, 7)
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
    elif type(y_1) == tuple:
        print('malloc', sys.getsizeof(y_1), 'y_1', [len(y_1)])
    else:
        print('malloc', sys.getsizeof(y_1), 'y_1')
    print(3, 8)
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
    elif type(q_1) == tuple:
        print('malloc', sys.getsizeof(q_1), 'q_1', [len(q_1)])
    else:
        print('malloc', sys.getsizeof(q_1), 'q_1')
    print(3, 9)
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
    elif type(r_1) == tuple:
        print('malloc', sys.getsizeof(r_1), 'r_1', [len(r_1)])
    else:
        print('malloc', sys.getsizeof(r_1), 'r_1')
    print(3, 10)
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
    elif type(q_1) == tuple:
        print('malloc', sys.getsizeof(q_1), 'q_1', [len(q_1)])
    else:
        print('malloc', sys.getsizeof(q_1), 'q_1')
    print(3, 11)
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
    elif type(w_1) == tuple:
        print('malloc', sys.getsizeof(w_1), 'w_1', [len(w_1)])
    else:
        print('malloc', sys.getsizeof(w_1), 'w_1')
    if instrument_read(w_1, 'w_1') < 0:
        print(4, 13)
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
        elif type(a_1) == tuple:
            print('malloc', sys.getsizeof(a_1), 'a_1', [len(a_1)])
        else:
            print('malloc', sys.getsizeof(a_1), 'a_1')
        print(4, 14)
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
        elif type(b_1) == tuple:
            print('malloc', sys.getsizeof(b_1), 'b_1', [len(b_1)])
        else:
            print('malloc', sys.getsizeof(b_1), 'b_1')
        print(4, 15)
        r_1 += instrument_read(a_1, 'a_1') + 3 * 2
        write_instrument_read(r_1, 'r_1')
    else:
        print(6, 17)
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
        elif type(a_1) == tuple:
            print('malloc', sys.getsizeof(a_1), 'a_1', [len(a_1)])
        else:
            print('malloc', sys.getsizeof(a_1), 'a_1')
        print(6, 18)
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
        elif type(b_1) == tuple:
            print('malloc', sys.getsizeof(b_1), 'b_1', [len(b_1)])
        else:
            print('malloc', sys.getsizeof(b_1), 'b_1')
    print(5, 19)
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
    elif type(z_1) == tuple:
        print('malloc', sys.getsizeof(z_1), 'z_1', [len(z_1)])
    else:
        print('malloc', sys.getsizeof(z_1), 'z_1')
    print(5, 20)
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
    elif type(q_1) == tuple:
        print('malloc', sys.getsizeof(q_1), 'q_1', [len(q_1)])
    else:
        print('malloc', sys.getsizeof(q_1), 'q_1')
    for i_1 in range(5):
        instrument_read(loop, 'loop').start_unroll
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
    print(1, 27)
    print(12, 28)
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
    elif type(a_2) == tuple:
        print('malloc', sys.getsizeof(a_2), 'a_2', [len(a_2)])
    else:
        print('malloc', sys.getsizeof(a_2), 'a_2')
    for i_2 in range(3):
        instrument_read(loop, 'loop').start_unroll
        print(14, 31)
        a_2 += instrument_read(i_2, 'i_2')
        write_instrument_read(a_2, 'a_2')
    instrument_read(loop, 'loop').stop_unroll
    print('exit scope 2')


if instrument_read(__name__, '__name__') == '__main__':
    for i_0 in range(1):
        main(2, 3)
        bruh()
