import sys
from instrument_lib import *
from loop import loop


def main(x1, y1):
    print('enter scope 1')
    print(1, 3)
    print(3, 4)
    q1 = 0.5 + instrument_read(x1, 'x1') * instrument_read(y1, 'y1') + 1 / 2
    write_instrument_read(q1, 'q1')
    print('malloc', sys.getsizeof(q1), 'q1')
    print(3, 5)
    r1 = instrument_read(x1, 'x1') + instrument_read(y1, 'y1')
    write_instrument_read(r1, 'r1')
    print('malloc', sys.getsizeof(r1), 'r1')
    print(3, 6)
    q1 = instrument_read(q1, 'q1') * instrument_read(r1, 'r1') - 3
    write_instrument_read(q1, 'q1')
    print('malloc', sys.getsizeof(q1), 'q1')
    print(3, 7)
    w1 = instrument_read(q1, 'q1') + instrument_read(r1, 'r1')
    write_instrument_read(w1, 'w1')
    print('malloc', sys.getsizeof(w1), 'w1')
    print('enter scope 2')
    if instrument_read(w1, 'w1') < 0:
        print(3, 8)
        print(4, 9)
        a2 = instrument_read(q1, 'q1') + 3
        write_instrument_read(a2, 'a2')
        print('malloc', sys.getsizeof(a2), 'a2')
        print(4, 10)
        b2 = instrument_read(a2, 'a2') * instrument_read(r1, 'r1')
        write_instrument_read(b2, 'b2')
        print('malloc', sys.getsizeof(b2), 'b2')
        print(4, 11)
        r1 += instrument_read(a2, 'a2') + 3 * 2
        write_instrument_read(r1, 'r1')
    else:
        print(3, 8)
        print(6, 13)
        a2 = instrument_read(q1, 'q1') - 3
        write_instrument_read(a2, 'a2')
        print('malloc', sys.getsizeof(a2), 'a2')
        print(6, 14)
        b2 = instrument_read(a2, 'a2') / instrument_read(r1, 'r1')
        write_instrument_read(b2, 'b2')
        print('malloc', sys.getsizeof(b2), 'b2')
    print('exit scope 2')
    print(5, 15)
    z1 = [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z1, 'z1')
    print('malloc', sys.getsizeof(z1), 'z1')
    print(5, 16)
    z1 += [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z1, 'z1')
    print(5, 17)
    r1, q1 = 2, 3
    write_instrument_read(q1, 'q1')
    print('malloc', sys.getsizeof(q1), 'q1')
    loop.start_unroll
    print('enter scope 3')
    for i3 in range(5):
        print(8, 20)
        z1[0][instrument_read(instrument_read(i3, 'i3'), 'i3')
            ] += instrument_read_sub(instrument_read_sub(instrument_read(z1,
            'z1'), 'z', 0, 'None', 'None', 'False'), 'z[0]', 
            instrument_read(i3, 'i3') + 1, 'None', 'None', 'False')
        write_instrument_read_sub(z1[0], 'z1[0]', instrument_read(
            instrument_read(i3, 'i3'), 'i3'), 'None', 'None', 'False')
    print('exit scope 3')
    loop.stop_unroll
    print('exit scope 1')


def bruh():
    print('enter scope 4')
    print(1, 23)
    print(12, 24)
    a4 = 1
    write_instrument_read(a4, 'a4')
    print('malloc', sys.getsizeof(a4), 'a4')
    print('enter scope 5')
    for i5 in range(3):
        print(14, 26)
        a4 += instrument_read(i5, 'i5')
        write_instrument_read(a4, 'a4')
    print('exit scope 5')
    print('exit scope 4')


print('enter scope 6')
if __name__ == '__main__':
    print(1, 28)
    main(2, 3)
    bruh()
else:
    print(1, 28)
print('exit scope 6')
