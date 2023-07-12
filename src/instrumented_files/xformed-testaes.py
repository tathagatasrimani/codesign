import sys
from instrument_lib import *
import sys
from instrument_lib import *


def main(a, b):
    print('enter scope 1')
    print(1, 3)
    a__1 = instrument_read(a, 'a')
    write_instrument_read(a__1, 'a__1')
    print('malloc', sys.getsizeof(a__1), 'a__1')
    b__1 = instrument_read(b, 'b')
    write_instrument_read(b__1, 'b__1')
    print('malloc', sys.getsizeof(b__1), 'b__1')
    print('exit scope 1')
    return instrument_read(a__1, 'a__1')
    print('exit scope 1')


print('enter scope 2')
if instrument_read(__name__, '__name__') == '__main__':
    print(6, 7)
    a__2 = 1
    write_instrument_read(a__2, 'a__2')
    print('malloc', sys.getsizeof(a__2), 'a__2')
    main(instrument_read(a__2, 'a__2'), b=a__2)
print('exit scope 2')
