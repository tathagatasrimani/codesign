
def add_nums(n, a):
    b = (a + 4) * 2
    c = b + a
    d = c * b
    if d > 5:
        d //= 2
    else:
        d += 3
    return d

if __name__ == "__main__":
    add_nums(5, 10)