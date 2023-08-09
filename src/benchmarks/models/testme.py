from loop import loop

def main(x, y):
    q = 0.5 + x * y + 1 / 2
    r = x + y
    q = q * r - 3
    w = q + r
    if w < 0:
        a = q + 3
        b = a * r
        r += a + 3 * 2
    else:
        a = q - 3
        b = a / r
    z = [[1, 2, 3, 4, 5, 6]]
    r, q = 2, 3
    loop.start_unroll
    for i in range(5):
        z[0][1] = 1
        z[0][i] += z[0][i+1]
    loop.stop_unroll

def bruh():
    a = 1
    loop.start_unroll
    for i in range(3):
        a += i
    loop.stop_unroll

if __name__ == "__main__":
    for i in range(1):
        loop().pattern_seek()
        main(2, 3)
        bruh()
