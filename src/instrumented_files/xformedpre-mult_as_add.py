import sys
from instrument_lib import *
from loop import loop
import numpy as np


def mult(a, b):
    print('enter scope 1')
    print(1, 5)
    print(3, 6)
    a_1 = a
    print(3, 7)
    b_1 = b
    print(3, 8)
    res_1 = 0
    for i_1 in range(a_1):
        print(5, 10)
        res_1 += b_1
    print('exit scope 1')
    return res_1
    print('exit scope 1')


if __name__ == '__main__':
    mult(2, 3)
