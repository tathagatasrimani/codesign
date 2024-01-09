import sys
from instrument_lib import *
import numpy as np
from loop import loop


def main():
    print('enter scope 1')
    print(1, 5)
    print(3, 6)
    a_1 = 2
    print(3, 7)
    b_1 = 3
    print(3, 8)
    c_1 = b_1 + a_1
    print(3, 9)
    d_1 = b_1 * a_1
    print('exit scope 1')


if __name__ == '__main__':
    main()
