import sys
from instrument_lib import *
from loop import loop
import numpy as np


def main(x, y):
    print('enter scope 1')
    print(1, 5)
    print(3, 6)
    x_1 = x
    print(3, 7)
    y_1 = y
    print(3, 8)
    q_1 = 0.5 + x_1 * y_1 + 1 / 2
    print(3, 9)
    r_1 = x_1 + y_1
    print(3, 10)
    q_1 = q_1 * r_1 - 3
    print(3, 11)
    w_1 = q_1 + r_1
    if w_1 < 0:
        print(4, 13)
        a_1 = q_1 + 3
        print(4, 14)
        b_1 = a_1 * r_1
        print(4, 15)
        r_1 += a_1 + 3 * 2
    else:
        print(6, 17)
        a_1 = q_1 - 3
        print(6, 18)
        b_1 = a_1 / r_1
    print(5, 19)
    z_1 = [[1, 2, 3, 4, 5, 6]]
    print(5, 20)
    r_1, q_1 = 2, 3
    loop.start_unroll
    for i_1 in range(5):
        print(8, 23)
        z_1[0][i_1] += z_1[0][i_1 + 1]
    loop.stop_unroll
    print('exit scope 1')


def bruh():
    print('enter scope 2')
    print(1, 27)
    print(12, 28)
    a_2 = 1
    loop.start_unroll
    for i_2 in range(3):
        print(14, 31)
        a_2 += i_2
    loop.stop_unroll
    print('exit scope 2')


if __name__ == '__main__':
    for i_0 in range(2):
        loop().pattern_seek()
        main(2, 3)
        bruh()
