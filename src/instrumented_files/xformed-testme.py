import sys
from instrument_lib import *
from memory import Memory
MEMORY_SIZE = 10000
memory_module = Memory(MEMORY_SIZE)
from loop import loop


def main(x, y):
    global memory_module
    print(1, 3)
    print(3, 4)
    q = 0.5 + instrument_read(x, 'x') * instrument_read(y, 'y') + 1 / 2
    write_instrument_read(q, 'q')
    memory_module.malloc('q', sys.getsizeof(q))
    print(memory_module.locations['q'].location, 'q', 'mem')
    print(3, 5)
    r = instrument_read(x, 'x') + instrument_read(y, 'y')
    write_instrument_read(r, 'r')
    memory_module.malloc('r', sys.getsizeof(r))
    print(memory_module.locations['r'].location, 'r', 'mem')
    print(3, 6)
    q = instrument_read(q, 'q') * instrument_read(r, 'r') - 3
    write_instrument_read(q, 'q')
    memory_module.malloc('q', sys.getsizeof(q))
    print(memory_module.locations['q'].location, 'q', 'mem')
    print(3, 7)
    w = instrument_read(q, 'q') + instrument_read(r, 'r')
    write_instrument_read(w, 'w')
    memory_module.malloc('w', sys.getsizeof(w))
    print(memory_module.locations['w'].location, 'w', 'mem')
    if instrument_read(w, 'w') < 0:
        print(3, 8)
        print(4, 9)
        a = instrument_read(q, 'q') + 3
        write_instrument_read(a, 'a')
        memory_module.malloc('a', sys.getsizeof(a))
        print(memory_module.locations['a'].location, 'a', 'mem')
        print(4, 10)
        b = instrument_read(a, 'a') * instrument_read(r, 'r')
        write_instrument_read(b, 'b')
        memory_module.malloc('b', sys.getsizeof(b))
        print(memory_module.locations['b'].location, 'b', 'mem')
        print(4, 11)
        r += instrument_read(a, 'a') + 3 * 2
    else:
        print(3, 8)
        print(6, 13)
        a = instrument_read(q, 'q') - 3
        write_instrument_read(a, 'a')
        memory_module.malloc('a', sys.getsizeof(a))
        print(memory_module.locations['a'].location, 'a', 'mem')
        print(6, 14)
        b = instrument_read(a, 'a') / instrument_read(r, 'r')
        write_instrument_read(b, 'b')
        memory_module.malloc('b', sys.getsizeof(b))
        print(memory_module.locations['b'].location, 'b', 'mem')
    print(5, 15)
    z = [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z, 'z')
    memory_module.malloc('z', sys.getsizeof(z))
    print(memory_module.locations['z'].location, 'z', 'mem')
    print(5, 16)
    z += [[1, 2, 3, 4, 5, 6]]
    instrument_read(loop, 'loop').start_unroll
    for i in range(5):
        print(8, 19)
        z[0][i] += instrument_read_sub(instrument_read_sub(instrument_read(
            z, 'z'), 'z', 0), 'z[0]', i + 1)
        write_instrument_read_sub(instrument_read_sub(instrument_read(z,
            'z'), 'z', 0), 'z[0]', i)
        print(8, 20)
        q = instrument_read_sub(instrument_read_sub(instrument_read(z, 'z'),
            'z', i), 'z[i]', i + 1)
        write_instrument_read(q, 'q')
        memory_module.malloc('q', sys.getsizeof(q))
        print(memory_module.locations['q'].location, 'q', 'mem')
    instrument_read(loop, 'loop').stop_unroll


def bruh():
    global memory_module
    print(1, 23)
    print(12, 24)
    a = 1
    write_instrument_read(a, 'a')
    memory_module.malloc('a', sys.getsizeof(a))
    print(memory_module.locations['a'].location, 'a', 'mem')
    instrument_read(loop, 'loop').start_unroll
    for i in range(3):
        print(14, 27)
        a += instrument_read(i, 'i')
    instrument_read(loop, 'loop').stop_unroll


if instrument_read(__name__, '__name__') == '__main__':
    print(1, 30)
    main(2, 3)
    bruh()
else:
    print(1, 30)
