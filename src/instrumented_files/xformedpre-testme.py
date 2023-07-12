import sys
from instrument_lib import *
from loop import loop


def main(x, y):
    print('enter scope 1')
    print(1, 3)
    x__1 = x
    y__1 = y
    print(3, 4)
    q__1 = 0.5 + x__1 * y__1 + 1 / 2
    print(3, 5)
    r__1 = x__1 + y__1
    print(3, 6)
    q__1 = q__1 * r__1 - 3
    print(3, 7)
    w__1 = q__1 + r__1
    print('enter scope 2')
    if w__1 < 0:
        print(4, 9)
        a__2 = q__1 + 3
        print(4, 10)
        b__2 = a__2 * r__1
        print(4, 11)
        r__1 += a__2 + 3 * 2
    else:
        print(6, 13)
        a__2 = q__1 - 3
        print(6, 14)
        b__2 = a__2 / r__1
    print('exit scope 2')
    print(5, 15)
    z__1 = [[1, 2, 3, 4, 5, 6]]
    print(5, 16)
    z__1 += [[1, 2, 3, 4, 5, 6]]
    print(5, 17)
    r__1, q__1 = 2, 3
    loop.start_unroll
    for i__1 in range(5):
        print('enter scope 3')
        print(8, 20)
        z__1[0][i__1] += z__1[0][i__1 + 1]
        print('exit scope 3')
    loop.stop_unroll
    print('exit scope 1')


def bruh():
    print('enter scope 4')
    print(1, 23)
    print(12, 24)
    a__4 = 1
    for i__4 in range(3):
        print('enter scope 5')
        print(14, 26)
        a__4 += i__4
        print('exit scope 5')
    print('exit scope 4')


print('enter scope 6')
if __name__ == '__main__':
    main(2, 3)
    bruh()
print('exit scope 6')
