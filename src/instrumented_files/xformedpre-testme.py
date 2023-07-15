import sys
from instrument_lib import *
from loop import loop


def main(x, y):
    print('enter scope 1')
    print(1, 3)
    x_1 = x
    y_1 = y
    print(3, 4)
    q_1 = 0.5 + x_1 * y_1 + 1 / 2
    print(3, 5)
    r_1 = x_1 + y_1
    print(3, 6)
    q_1 = q_1 * r_1 - 3
    print(3, 7)
    w_1 = q_1 + r_1
    if w_1 < 0:
        print(4, 9)
        a_1 = q_1 + 3
        print(4, 10)
        b_1 = a_1 * r_1
        print(4, 11)
        r_1 += a_1 + 3 * 2
    else:
        print(6, 13)
        a_1 = q_1 - 3
        print(6, 14)
        b_1 = a_1 / r_1
    print(5, 15)
    z_1 = [[1, 2, 3, 4, 5, 6]]
    print(5, 16)
    z_1 += [[1, 2, 3, 4, 5, 6]]
    print(5, 17)
    r_1, q_1 = 2, 3
    for i_1 in range(5):
        print(8, 19)
        z_1[0][i_1] += z_1[0][i_1 + 1]
    print('exit scope 1')


def bruh():
    print('enter scope 2')
    print(1, 21)
    print(12, 22)
    a_2 = 1
    for i_2 in range(3):
        print(14, 24)
        a_2 += i_2
    print('exit scope 2')


if __name__ == '__main__':
    main(2, 3)
    bruh()
