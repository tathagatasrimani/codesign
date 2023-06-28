import sys
from loop import loop


def main(x, y):
    print(1, 3)
    print(3, 4)
    q = 0.5 + x * y + 1 / 2
    print(3, 5)
    r = x + y
    print(3, 6)
    q = q * r - 3
    print(3, 7)
    w = q + r
    if w < 0:
        print(3, 8)
        print(4, 9)
        a = q + 3
        print(4, 10)
        b = a * r
        print(4, 11)
        r += a + 3 * 2
    else:
        print(3, 8)
        print(6, 13)
        a = q - 3
        print(6, 14)
        b = a / r
    print(5, 15)
    z = [1, 2, 3, 4, 5]
    print(5, 16)
    z += [1, 2, 3, 4, 5]
    loop.start_unroll
    for i in range(5):
        print(8, 19)
        z[i] += 1
    loop.stop_unroll


def bruh():
    print(1, 22)
    print(12, 23)
    a = 1
    loop.start_unroll
    for i in range(3):
        print(14, 26)
        a += i
    loop.stop_unroll


if __name__ == '__main__':
    print(1, 29)
    main(2, 3)
    bruh()
else:
    print(1, 29)
