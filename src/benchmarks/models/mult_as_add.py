from loop import loop

def mult(a, b):
    res = 0
    loop.start_unroll
    for i in range(a):
        res += b
    loop.stop_unroll
    return res

if __name__ == "__main__":
    mult(2, 3)