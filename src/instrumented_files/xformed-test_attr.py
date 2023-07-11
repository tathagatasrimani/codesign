import sys
from instrument_lib import *
def main():
    print('enter scope 1')
    print(1, 2)
    print(3, 3)
    b__1 = 2
    write_instrument_read(b__1, 'b__1')
    print('malloc', sys.getsizeof(b__1), 'b__1')
    print(3, 4)
    b__1 = 3
    write_instrument_read(b__1, 'b__1')
    print('malloc', sys.getsizeof(b__1), 'b__1')
    print(3, 5)
    e__1 = instrument_read(b__1, 'b__1')
    write_instrument_read(e__1, 'e__1')
    print('malloc', sys.getsizeof(e__1), 'e__1')
    print('exit scope 1')


print('enter scope 2')
if __name__ == '__main__':
    print(1, 7)
    main()
else:
    print(1, 7)
print('exit scope 2')
