def cool_func(num):
    return num + 2 // 3

def other_func(f):
    a = 1 + 1
    b = 3 + 4 * 5
    return a - b

def do_something(n, a):
    b = (a + 4) * 2
    c = b + a
    d = c * b
    e = other_func(d) / 2
    if d > 5:
        d //= 2
    else:
        d += 1 * 3 + 4 * 2
    return d + e

if __name__ == "__main__":
    do_something(5, 10)