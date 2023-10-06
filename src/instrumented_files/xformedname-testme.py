from loop import loop
import numpy as np


def main(x, y):
    x_1 = x
    y_1 = y
    q_1 = 0.5 + x_1 * y_1 + 1 / 2
    r_1 = x_1 + y_1
    q_1 = q_1 * r_1 - 3
    w_1 = q_1 + r_1
    if w_1 < 0:
        a_1 = q_1 + 3
        b_1 = a_1 * r_1
        r_1 += a_1 + 3 * 2
    else:
        a_1 = q_1 - 3
        b_1 = a_1 / r_1
    z_1 = [[1, 2, 3, 4, 5, 6]]
    r_1, q_1 = 2, 3
    for i_1 in range(5):
        loop.start_unroll
        z_1[0][i_1] += z_1[0][i_1 + 1]
    loop.stop_unroll


def bruh():
    a_2 = 1
    for i_2 in range(3):
        loop.start_unroll
        a_2 += i_2
    loop.stop_unroll


if __name__ == '__main__':
    for i_0 in range(1):
        main(2, 3)
        bruh()
