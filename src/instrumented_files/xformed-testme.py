import sys
from memory import Memory
MEMORY_SIZE = 10000
from loop import loop


def main(x, y):
    print(1, 3)
    memory_module = Memory(MEMORY_SIZE)
    print(3, 4)
    q = 0.5 + x * y + 1 / 2
    memory_module.malloc('q', sys.getsizeof(q))
    print(memory_module.locations['q'].location, 'q', 'mem')
    print(3, 5)
    r = x + y
    memory_module.malloc('r', sys.getsizeof(r))
    print(memory_module.locations['r'].location, 'r', 'mem')
    print(3, 6)
    q = q * r - 3
    memory_module.malloc('q', sys.getsizeof(q))
    print(memory_module.locations['q'].location, 'q', 'mem')
    print(3, 7)
    w = q + r
    memory_module.malloc('w', sys.getsizeof(w))
    print(memory_module.locations['w'].location, 'w', 'mem')
    if w < 0:
        print(3, 8)
        print(4, 9)
        a = q + 3
        memory_module.malloc('a', sys.getsizeof(a))
        print(memory_module.locations['a'].location, 'a', 'mem')
        print(4, 10)
        b = a * r
        memory_module.malloc('b', sys.getsizeof(b))
        print(memory_module.locations['b'].location, 'b', 'mem')
        print(4, 11)
        r += a + 3 * 2
    else:
        print(3, 8)
        print(6, 13)
        a = q - 3
        memory_module.malloc('a', sys.getsizeof(a))
        print(memory_module.locations['a'].location, 'a', 'mem')
        print(6, 14)
        b = a / r
        memory_module.malloc('b', sys.getsizeof(b))
        print(memory_module.locations['b'].location, 'b', 'mem')
    print(5, 15)
    z = [1, 2, 3, 4, 5]
    memory_module.malloc('z', sys.getsizeof(z))
    print(memory_module.locations['z'].location, 'z', 'mem')
    print(5, 16)
    z += [1, 2, 3, 4, 5]
    loop.start_unroll
    for i in range(5):
        print(8, 19)
        z[i] += 1
    loop.stop_unroll


def bruh():
    print(1, 22)
    memory_module = Memory(MEMORY_SIZE)
    print(12, 23)
    a = 1
    memory_module.malloc('a', sys.getsizeof(a))
    print(memory_module.locations['a'].location, 'a', 'mem')
    loop.start_unroll
    for i in range(3):
        print(14, 26)
        a += i
    loop.stop_unroll


if __name__ == '__main__':
    print(1, 29)
    main(2, 3)
    bruh()
else:
    print(1, 29)
