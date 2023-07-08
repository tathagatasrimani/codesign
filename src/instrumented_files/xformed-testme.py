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
    print('malloc', id(q), sys.getsizeof(q))
    memory_module.malloc('id(q)', sys.getsizeof(q))
    print(memory_module.locations['id(q)'].location, 'q', 'mem')
    print(3, 5)
    r = instrument_read(x, 'x') + instrument_read(y, 'y')
    write_instrument_read(r, 'r')
    print('malloc', id(r), sys.getsizeof(r))
    memory_module.malloc('id(r)', sys.getsizeof(r))
    print(memory_module.locations['id(r)'].location, 'r', 'mem')
    print(3, 6)
    q = instrument_read(q, 'q') * instrument_read(r, 'r') - 3
    write_instrument_read(q, 'q')
    print('malloc', id(q), sys.getsizeof(q))
    memory_module.malloc('id(q)', sys.getsizeof(q))
    print(memory_module.locations['id(q)'].location, 'q', 'mem')
    print(3, 7)
    w = instrument_read(q, 'q') + instrument_read(r, 'r')
    write_instrument_read(w, 'w')
    print('malloc', id(w), sys.getsizeof(w))
    memory_module.malloc('id(w)', sys.getsizeof(w))
    print(memory_module.locations['id(w)'].location, 'w', 'mem')
    if instrument_read(w, 'w') < 0:
        print(3, 8)
        print(4, 9)
        a = instrument_read(q, 'q') + 3
        write_instrument_read(a, 'a')
        print('malloc', id(a), sys.getsizeof(a))
        memory_module.malloc('id(a)', sys.getsizeof(a))
        print(memory_module.locations['id(a)'].location, 'a', 'mem')
        print(4, 10)
        b = instrument_read(a, 'a') * instrument_read(r, 'r')
        write_instrument_read(b, 'b')
        print('malloc', id(b), sys.getsizeof(b))
        memory_module.malloc('id(b)', sys.getsizeof(b))
        print(memory_module.locations['id(b)'].location, 'b', 'mem')
        print(4, 11)
        r += instrument_read(a, 'a') + 3 * 2
        write_instrument_read(r, 'r')
    else:
        print(3, 8)
        print(6, 13)
        a = instrument_read(q, 'q') - 3
        write_instrument_read(a, 'a')
        print('malloc', id(a), sys.getsizeof(a))
        memory_module.malloc('id(a)', sys.getsizeof(a))
        print(memory_module.locations['id(a)'].location, 'a', 'mem')
        print(6, 14)
        b = instrument_read(a, 'a') / instrument_read(r, 'r')
        write_instrument_read(b, 'b')
        print('malloc', id(b), sys.getsizeof(b))
        memory_module.malloc('id(b)', sys.getsizeof(b))
        print(memory_module.locations['id(b)'].location, 'b', 'mem')
    print(5, 15)
    z = [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z, 'z')
    print('malloc', id(z), sys.getsizeof(z))
    memory_module.malloc('id(z)', sys.getsizeof(z))
    print(memory_module.locations['id(z)'].location, 'z', 'mem')
    print(5, 16)
    z += [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z, 'z')
    print(5, 17)
    r, q = 2, 3
    write_instrument_read(q, 'q')
    print('malloc', id(q), sys.getsizeof(q))
    memory_module.malloc('id(q)', sys.getsizeof(q))
    print(memory_module.locations['id(q)'].location, 'q', 'mem')
    instrument_read(loop, 'loop').start_unroll
    for i in range(5):
        print(8, 20)
        z[0][i] += instrument_read_sub(instrument_read_sub(instrument_read(
            z, 'z'), 'z', 0), 'z[0]', i + 1)
        write_instrument_read_sub(instrument_read_sub(instrument_read(z,
            'z'), 'z', 0), 'z[0]', i)
    instrument_read(loop, 'loop').stop_unroll


def bruh():
    global memory_module
    print(1, 23)
    print(12, 24)
    a = 1
    write_instrument_read(a, 'a')
    print('malloc', id(a), sys.getsizeof(a))
    memory_module.malloc('id(a)', sys.getsizeof(a))
    print(memory_module.locations['id(a)'].location, 'a', 'mem')
    for i in range(3):
        print(14, 26)
        a += instrument_read(i, 'i')
        write_instrument_read(a, 'a')


if instrument_read(__name__, '__name__') == '__main__':
    print(1, 28)
    main(2, 3)
    bruh()
else:
    print(1, 28)
