import sys
from instrument_lib import *
def main(a, b):
    print('enter scope 1')
    print(1, 3)
    a__1 = a
    b__1 = b
    print('exit scope 1')
    return a__1
    print('exit scope 1')


print('enter scope 2')
if __name__ == '__main__':
    print(6, 7)
    a__2 = 1
    main(a__2, b=a__2)
print('exit scope 2')
