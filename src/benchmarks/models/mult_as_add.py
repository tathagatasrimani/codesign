import numpy as np
def mult(a, b):
    res = 0
    for i in range(a):
        res += b
    return res

if __name__ == "__main__":
    mult(2, 3)