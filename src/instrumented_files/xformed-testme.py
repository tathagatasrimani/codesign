import sys
from instrument_lib import *
from loop import loop


def main(x0, y0):
    print('enter scope 0')
    print(1, 3)
    print(3, 4)
    q0 = 0.5 + instrument_read(x0, 'x0') * instrument_read(y0, 'y0') + 1 / 2
    write_instrument_read(q0, 'q0')
    print('malloc', sys.getsizeof(q0), 'q0')
    print(3, 5)
    r0 = instrument_read(x0, 'x0') + instrument_read(y0, 'y0')
    write_instrument_read(r0, 'r0')
    print('malloc', sys.getsizeof(r0), 'r0')
    print(3, 6)
    q0 = instrument_read(q0, 'q0') * instrument_read(r0, 'r0') - 3
    write_instrument_read(q0, 'q0')
    print('malloc', sys.getsizeof(q0), 'q0')
    print(3, 7)
    w0 = instrument_read(q0, 'q0') + instrument_read(r0, 'r0')
    write_instrument_read(w0, 'w0')
    print('malloc', sys.getsizeof(w0), 'w0')
    print('enter scope 1')
    if instrument_read(w0, 'w0') < 0:
        print(3, 8)
        print(4, 9)
        a1 = instrument_read(q0, 'q0') + 3
        write_instrument_read(a1, 'a1')
        print('malloc', sys.getsizeof(a1), 'a1')
        print(4, 10)
        b1 = instrument_read(a1, 'a1') * instrument_read(r0, 'r0')
        write_instrument_read(b1, 'b1')
        print('malloc', sys.getsizeof(b1), 'b1')
        print(4, 11)
        r0 += instrument_read(a1, 'a1') + 3 * 2
        write_instrument_read(r0, 'r0')
    else:
        print(3, 8)
        print(6, 13)
        a1 = instrument_read(q0, 'q0') - 3
        write_instrument_read(a1, 'a1')
        print('malloc', sys.getsizeof(a1), 'a1')
        print(6, 14)
        b1 = instrument_read(a1, 'a1') / instrument_read(r0, 'r0')
        write_instrument_read(b1, 'b1')
        print('malloc', sys.getsizeof(b1), 'b1')
    print('exit scope 1')
    print(5, 15)
    z0 = [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z0, 'z0')
    print('malloc', sys.getsizeof(z0), 'z0')
    print(5, 16)
    z0 += [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z0, 'z0')
    print(5, 17)
    r0, q0 = 2, 3
    write_instrument_read(q0, 'q0')
    print('malloc', sys.getsizeof(q0), 'q0')
    loop.start_unroll
    print('enter scope 2')
    for i2 in range(5):
        z0[0][instrument_read(i2, 'i2')] += instrument_read_sub(
            instrument_read_sub(instrument_read(z0, 'z0'), 'z', 0), 'z[0]',
            instrument_read(i2, 'i2') + 1)
    print('exit scope 2')
    loop.stop_unroll
    print('exit scope 0')


def bruh():
    print('enter scope 3')
    print(1, 23)
    print(12, 24)
    a3 = 1
    write_instrument_read(a3, 'a3')
    print('malloc', sys.getsizeof(a3), 'a3')
    print('enter scope 4')
    for i4 in range(3):
        a3 += instrument_read(i4, 'i4')
    print('exit scope 4')
    print('exit scope 3')


print('enter scope 5')
if __name__ == '__main__':
    print(1, 28)
    main(2, 3)
    bruh()
else:
    print(1, 28)
print('exit scope 5')
