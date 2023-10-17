import loop as loop
import numpy as np
def main():
    a = 3
    b = 2
    c = 0
    d = 0
    for i in range(10):
        for j in range(10):
            c += a * b
            d *= a + b
    return c


if __name__ == "__main__":
    main()