import sys
from instrument_lib import *
def cool_func(num):
    print(1, 1)
    return num + 2 // 3


def other_func(f):
    print(1, 4)
    print(7, 5)
    a = 1 + 1
    print(7, 6)
    b = 3 + 4 * 5
    return a - b


def do_something(n, a):
    print(1, 9)
    print(11, 10)
    b = (a + 4) * 2
    print(11, 11)
    c = b + a
    print(11, 12)
    d = c * b
    print(11, 13)
    e = other_func(d) / 2
    if d > 5:
        print(11, 14)
        print(12, 15)
        d //= 2
    else:
        print(11, 14)
        print(14, 17)
        d += 1 * 3 + 4 * 2
    return d + e


if __name__ == '__main__':
    print(1, 20)
    do_something(5, 10)
else:
    print(1, 20)
