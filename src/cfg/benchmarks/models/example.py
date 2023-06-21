def main(x, y):
    q = 0.5 + x * y + 1 / 2
    r = x + y
    q = q * r - 3
    w = q + r
    if w > 0:
        a = q + 3
        b = a * r
        r += a + 3 * 2
    else:
        a = q - 3
        b = a / r
if __name__ == "__main__":
    main(2, 3)