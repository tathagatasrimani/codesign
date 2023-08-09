import sys
from instrument_lib import *
import sys
from instrument_lib import *
from loop import loop


def main(x, y):
    print('enter scope 1')
    print(1, 3)
    x_1 = instrument_read(x, 'x')
    write_instrument_read(x_1, 'x_1')
    print('malloc', sys.getsizeof(x_1), 'x_1')
    y_1 = instrument_read(y, 'y')
    write_instrument_read(y_1, 'y_1')
    print('malloc', sys.getsizeof(y_1), 'y_1')
    print(3, 4)
    q_1 = 0.5 + instrument_read(x_1, 'x_1') * instrument_read(y_1, 'y_1'
        ) + 1 / 2
    write_instrument_read(q_1, 'q_1')
    print('malloc', sys.getsizeof(q_1), 'q_1')
    print(3, 5)
    r_1 = instrument_read(x_1, 'x_1') + instrument_read(y_1, 'y_1')
    write_instrument_read(r_1, 'r_1')
    print('malloc', sys.getsizeof(r_1), 'r_1')
    print(3, 6)
    q_1 = instrument_read(q_1, 'q_1') * instrument_read(r_1, 'r_1') - 3
    write_instrument_read(q_1, 'q_1')
    print('malloc', sys.getsizeof(q_1), 'q_1')
    print(3, 7)
    w_1 = instrument_read(q_1, 'q_1') + instrument_read(r_1, 'r_1')
    write_instrument_read(w_1, 'w_1')
    print('malloc', sys.getsizeof(w_1), 'w_1')
    if instrument_read(w_1, 'w_1') < 0:
        print(4, 9)
        a_1 = instrument_read(q_1, 'q_1') + 3
        write_instrument_read(a_1, 'a_1')
        print('malloc', sys.getsizeof(a_1), 'a_1')
        print(4, 10)
        b_1 = instrument_read(a_1, 'a_1') * instrument_read(r_1, 'r_1')
        write_instrument_read(b_1, 'b_1')
        print('malloc', sys.getsizeof(b_1), 'b_1')
        print(4, 11)
        r_1 += instrument_read(a_1, 'a_1') + 3 * 2
        write_instrument_read(r_1, 'r_1')
    else:
        print(6, 13)
        a_1 = instrument_read(q_1, 'q_1') - 3
        write_instrument_read(a_1, 'a_1')
        print('malloc', sys.getsizeof(a_1), 'a_1')
        print(6, 14)
        b_1 = instrument_read(a_1, 'a_1') / instrument_read(r_1, 'r_1')
        write_instrument_read(b_1, 'b_1')
        print('malloc', sys.getsizeof(b_1), 'b_1')
    print(5, 15)
    z_1 = [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z_1, 'z_1')
    print('malloc', sys.getsizeof(z_1), 'z_1')
    print(5, 16)
    r_1, q_1 = 2, 3
    write_instrument_read(q_1, 'q_1')
    print('malloc', sys.getsizeof(q_1), 'q_1')
    instrument_read(loop, 'loop').start_unroll
    for i_1 in range(5):
        print(8, 19)
        z_1[0][1] = 1
        write_instrument_read_sub(z_1[0], 'z_1[0]', 1, None, None, False)
        print(8, 20)
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
    print(1, 23)
    print(12, 24)
    a_2 = 1
    write_instrument_read(a_2, 'a_2')
    print('malloc', sys.getsizeof(a_2), 'a_2')
    instrument_read(loop, 'loop').start_unroll
    for i_2 in range(3):
        print(14, 27)
        a_2 += instrument_read(i_2, 'i_2')
        write_instrument_read(a_2, 'a_2')
    instrument_read(loop, 'loop').stop_unroll
    print('exit scope 2')


if instrument_read(__name__, '__name__') == '__main__':
    for i_0 in range(1):
        loop().pattern_seek()
        main(2, 3)
        bruh()
