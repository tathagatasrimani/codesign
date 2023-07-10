import sys
from instrument_lib import *
from loop import loop


def main(x, y):
    print('enter scope 1')
    print(1, 3)
    print(3, 4)
    q = 0.5 + instrument_read(x, 'x') * instrument_read(y, 'y') + 1 / 2
    write_instrument_read(q, 'q')
    print('malloc', id(q), sys.getsizeof(q), 'q')
    print(3, 5)
    r = instrument_read(x, 'x') + instrument_read(y, 'y')
    write_instrument_read(r, 'r')
    print('malloc', id(r), sys.getsizeof(r), 'r')
    print(3, 6)
    q = instrument_read(q, 'q') * instrument_read(r, 'r') - 3
    write_instrument_read(q, 'q')
    print('malloc', id(q), sys.getsizeof(q), 'q')
    print(3, 7)
    w = instrument_read(q, 'q') + instrument_read(r, 'r')
    write_instrument_read(w, 'w')
    print('malloc', id(w), sys.getsizeof(w), 'w')
    print('enter scope 0')
    if instrument_read(w, 'w') < 0:
        print(3, 8)
        print(4, 9)
        a = instrument_read(q, 'q') + 3
        write_instrument_read(a, 'a')
        print('malloc', id(a), sys.getsizeof(a), 'a')
        print(4, 10)
        b = instrument_read(a, 'a') * instrument_read(r, 'r')
        write_instrument_read(b, 'b')
        print('malloc', id(b), sys.getsizeof(b), 'b')
        print(4, 11)
        r += instrument_read(a, 'a') + 3 * 2
        write_instrument_read(r, 'r')
    else:
        print(3, 8)
        print(6, 13)
        a = instrument_read(q, 'q') - 3
        write_instrument_read(a, 'a')
        print('malloc', id(a), sys.getsizeof(a), 'a')
        print(6, 14)
        b = instrument_read(a, 'a') / instrument_read(r, 'r')
        write_instrument_read(b, 'b')
        print('malloc', id(b), sys.getsizeof(b), 'b')
    print('exit scope 0')
    print(5, 15)
    z = [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z, 'z')
    print('malloc', id(z), sys.getsizeof(z), 'z')
    print(5, 16)
    z += [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z, 'z')
    print(5, 17)
    r, q = 2, 3
    write_instrument_read(q, 'q')
    print('malloc', id(q), sys.getsizeof(q), 'q')
    instrument_read(loop, 'loop').start_unroll
    for i in range(5):
        print(8, 20)
        z[0][i] += instrument_read_sub(instrument_read_sub(instrument_read(
            z, 'z'), 'z', 0), 'z[0]', i + 1)
        write_instrument_read_sub(instrument_read_sub(instrument_read(z,
            'z'), 'z', 0), 'z[0]', i)
    instrument_read(loop, 'loop').stop_unroll
    print('exit scope 1')


def bruh():
    print('enter scope 2')
    print(1, 23)
    print(12, 24)
    a = 1
    write_instrument_read(a, 'a')
    print('malloc', id(a), sys.getsizeof(a), 'a')
    for i in range(3):
        print(14, 26)
        a += instrument_read(i, 'i')
        write_instrument_read(a, 'a')
    print('exit scope 2')


print('enter scope 3')
if instrument_read(__name__, '__name__') == '__main__':
    print(1, 28)
    main(2, 3)
    bruh()
else:
    print(1, 28)
print('exit scope 3')
