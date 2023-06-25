import sys
def bruh():
    print(1, 2)
    return 3, 4


def main():
    print(1, 5)
    print(7, 6)
    i, j = 1, 2
    print(7, 7)
    b = [1, 2, 3]
    print(7, 8)
    k, l = bruh()
    c: int
    d: int = 5
    print(7, 11)
    b[0] = 2
    if 3 < 4 < 5 > 6:
        print(7, 12)
        print(8, 13)
        c = 2
    else:
        print(7, 12)


if __name__ == '__main__':
    print(1, 15)
    main()
else:
    print(1, 15)
