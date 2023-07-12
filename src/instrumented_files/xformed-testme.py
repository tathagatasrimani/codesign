import sys
from instrument_lib import *
import sys
from instrument_lib import *
from loop import loop


def main(x, y):
    print('enter scope 1')
    print(1, 3)
    x__1 = instrument_read(x, 'x')
    write_instrument_read(x__1, 'x__1')
    print('malloc', sys.getsizeof(x__1), 'x__1')
    y__1 = instrument_read(y, 'y')
    write_instrument_read(y__1, 'y__1')
    print('malloc', sys.getsizeof(y__1), 'y__1')
    print(3, 4)
    q__1 = 0.5 + instrument_read(x__1, 'x__1') * instrument_read(y__1, 'y__1'
        ) + 1 / 2
    write_instrument_read(q__1, 'q__1')
    print('malloc', sys.getsizeof(q__1), 'q__1')
    print(3, 5)
    r__1 = instrument_read(x__1, 'x__1') + instrument_read(y__1, 'y__1')
    write_instrument_read(r__1, 'r__1')
    print('malloc', sys.getsizeof(r__1), 'r__1')
    print(3, 6)
    q__1 = instrument_read(q__1, 'q__1') * instrument_read(r__1, 'r__1') - 3
    write_instrument_read(q__1, 'q__1')
    print('malloc', sys.getsizeof(q__1), 'q__1')
    print(3, 7)
    w__1 = instrument_read(q__1, 'q__1') + instrument_read(r__1, 'r__1')
    write_instrument_read(w__1, 'w__1')
    print('malloc', sys.getsizeof(w__1), 'w__1')
    print('enter scope 2')
    if instrument_read(w__1, 'w__1') < 0:
        print(4, 9)
        a__2 = instrument_read(q__1, 'q__1') + 3
        write_instrument_read(a__2, 'a__2')
        print('malloc', sys.getsizeof(a__2), 'a__2')
        print(4, 10)
        b__2 = instrument_read(a__2, 'a__2') * instrument_read(r__1, 'r__1')
        write_instrument_read(b__2, 'b__2')
        print('malloc', sys.getsizeof(b__2), 'b__2')
        print(4, 11)
        r__1 += instrument_read(a__2, 'a__2') + 3 * 2
        write_instrument_read(r__1, 'r__1')
    else:
        print(6, 13)
        a__2 = instrument_read(q__1, 'q__1') - 3
        write_instrument_read(a__2, 'a__2')
        print('malloc', sys.getsizeof(a__2), 'a__2')
        print(6, 14)
        b__2 = instrument_read(a__2, 'a__2') / instrument_read(r__1, 'r__1')
        write_instrument_read(b__2, 'b__2')
        print('malloc', sys.getsizeof(b__2), 'b__2')
    print('exit scope 2')
    print(5, 15)
    z__1 = [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z__1, 'z__1')
    print('malloc', sys.getsizeof(z__1), 'z__1')
    print(5, 16)
    z__1 += [[1, 2, 3, 4, 5, 6]]
    write_instrument_read(z__1, 'z__1')
    print(5, 17)
    r__1, q__1 = 2, 3
    write_instrument_read(q__1, 'q__1')
    print('malloc', sys.getsizeof(q__1), 'q__1')
    instrument_read(loop, 'loop').start_unroll
    for i__1 in range(5):
        print('enter scope 3')
        print(8, 20)
        z__1[0][instrument_read(i__1, 'i__1')] += instrument_read_sub(
            instrument_read_sub(instrument_read(z__1, 'z__1'), 'z__1', 0,
            None, None, False), 'z__1[0]', instrument_read(i__1, 'i__1') + 
            1, None, None, False)
        write_instrument_read_sub(z__1[0], 'z__1[0]', instrument_read(
            instrument_read(i__1, 'i__1'), 'i__1'), None, None, False)
        print('exit scope 3')
    instrument_read(loop, 'loop').stop_unroll
    print('exit scope 1')


def bruh():
    print('enter scope 4')
    print(1, 23)
    print(12, 24)
    a__4 = 1
    write_instrument_read(a__4, 'a__4')
    print('malloc', sys.getsizeof(a__4), 'a__4')
    for i__4 in range(3):
        print('enter scope 5')
        print(14, 26)
        a__4 += instrument_read(i__4, 'i__4')
        write_instrument_read(a__4, 'a__4')
        print('exit scope 5')
    print('exit scope 4')


print('enter scope 6')
if instrument_read(__name__, '__name__') == '__main__':
    main(2, 3)
    bruh()
print('exit scope 6')
