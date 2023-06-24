import sys
def main(x, y):
    print(1, 1)
    print(3, 2)
    q = 0.5 + x * y + 1 / 2
    print(3, 3)
    r = x + y
    print(3, 4)
    q = q * r - 3
    print(3, 5)
    w = q + r
    if w < 0:
        print(3, 6)
        print(4, 7)
        a = q + 3
        print(4, 8)
        b = a * r
        print(4, 9)
        r += a + 3 * 2
    else:
        print(3, 6)
        print(6, 11)
        a = q - 3
        print(6, 12)
        b = a / r


if __name__ == '__main__':
    print(1, 13)
    main(2, 3)
else:
    print(1, 13)
