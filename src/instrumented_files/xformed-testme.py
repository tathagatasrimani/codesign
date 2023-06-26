import sys
def main(x, y):
    print(1, 2)
    print(3, 3)
    q = 0.5 + x * y + 1 / 2
    print(3, 4)
    r = x + y
    print(3, 5)
    q = q * r - 3
    print(3, 6)
    w = q + r
    if w < 0:
        print(3, 7)
        print(4, 8)
        a = q + 3
        print(4, 9)
        b = a * r
        print(4, 10)
        r += a + 3 * 2
    else:
        print(3, 7)
        print(6, 12)
        a = q - 3
        print(6, 13)
        b = a / r
    print(5, 14)
    z = [1, 2, 3, 4, 5]
    print(5, 15)
    z += [1, 2, 3, 4, 5]
    print(7, 16)
    for i in range(5):
        print(8, 17)
        z[i] += 1


if __name__ == '__main__':
    print(1, 18)
    main(2, 3)
else:
    print(1, 18)
