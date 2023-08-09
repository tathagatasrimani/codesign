import sys
from instrument_lib import *
from loop import loop
import numpy as np


def main(x, y):
    print('enter scope 1')
    print(1, 4)
    x_1 = x
    y_1 = y
    print(3, 5)
    q_1 = 0.5 + x_1 * y_1 + 1 / 2
    print(3, 6)
    r_1 = x_1 + y_1
    print(3, 7)
    q_1 = q_1 * r_1 - 3
    print(3, 8)
    w_1 = q_1 + r_1
    if w_1 < 0:
        print(4, 10)
        a_1 = q_1 + 3
        print(4, 11)
        b_1 = a_1 * r_1
        print(4, 12)
        r_1 += a_1 + 3 * 2
    else:
        print(6, 14)
        a_1 = q_1 - 3
        print(6, 15)
        b_1 = a_1 / r_1
    print(5, 16)
    z_1 = [[1, 2, 3, 4, 5, 6]]
    print(5, 17)
    d_1 = [[[1], [1]], [[1], [1]]]
    print(5, 18)
    r_1, q_1 = 2, 3
    print(5, 19)
    g_1 = np.zeros((5, 4, 2))
    loop.start_unroll
    for i_1 in range(5):
        print(8, 22)
        z_1[0][1] = 1
        print(8, 23)
        z_1[0][i_1] += z_1[0][i_1 + 1]
    loop.stop_unroll
    print('exit scope 1')


def bruh():
    print('enter scope 2')
    print(1, 26)
    print(12, 27)
    a_2 = 1
    loop.start_unroll
    for i_2 in range(3):
        print(14, 30)
        a_2 += i_2
    loop.stop_unroll
    print('exit scope 2')


if __name__ == '__main__':
    for i_0 in range(1):
        loop().pattern_seek()
        main(2, 3)
        bruh()
