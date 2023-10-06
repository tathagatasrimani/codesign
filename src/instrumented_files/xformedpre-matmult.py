import sys
from instrument_lib import *
from loop import loop
import numpy as np


def main():
    print('enter scope 1')
    print(1, 5)
    print(3, 6)
    a_1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    print(3, 7)
    b_1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    print(3, 8)
    c_1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i_1 in range(3):
        for j_1 in range(3):
            loop.start_unroll
            for k_1 in range(3):
                print(10, 13)
                c_1[i_1][j_1] += a_1[i_1][k_1] * b_1[k_1][j_1]
            loop.stop_unroll
    print('exit scope 1')


if __name__ == '__main__':
    main()
