import loop as loop
import numpy as np


def main():
    a_1 = 3
    b_1 = 2
    c_1 = 0
    d_1 = 0
    for i_1 in range(10):
        for j_1 in range(10):
            c_1 += a_1 * b_1
            d_1 *= a_1 + b_1
    return c_1


if __name__ == '__main__':
    main()
