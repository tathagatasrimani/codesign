import sys
from instrument_lib import *
import loop as loop
import numpy as np


def main():
    print('enter scope 1')
    print(1, 5)
    print(3, 6)
    a_1 = 3
    print(3, 7)
    b_1 = 2
    print(3, 8)
    c_1 = 0
    print(3, 9)
    d_1 = 0
    for i_1 in range(10):
        for j_1 in range(10):
            print(7, 12)
            c_1 += a_1 * b_1
            print(7, 13)
            d_1 *= a_1 + b_1
    print('exit scope 1')
    return c_1
    print('exit scope 1')


if __name__ == '__main__':
    main()
