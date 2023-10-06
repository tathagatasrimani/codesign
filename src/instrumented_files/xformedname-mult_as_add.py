from loop import loop
import numpy as np


def mult(a, b):
    a_1 = a
    b_1 = b
    res_1 = 0
    for i_1 in range(a_1):
        res_1 += b_1
    return res_1


if __name__ == '__main__':
    mult(2, 3)
