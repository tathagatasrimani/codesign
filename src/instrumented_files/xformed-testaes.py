import sys
from instrument_lib import *
from memory import Memory
MEMORY_SIZE = 10000
memory_module = Memory(MEMORY_SIZE)
if instrument_read(__name__, '__name__') == '__main__':
    print(1, 3)
    print(2, 4)
    matrix = [[0]]
    write_instrument_read(matrix, 'matrix')
    memory_module.malloc('matrix', sys.getsizeof(matrix))
    print(memory_module.locations['matrix'].location, 'matrix', 'mem')
    for i in range(1):
        for j in range(1):
            print(7, 7)
            a = instrument_read_sub(instrument_read_sub(instrument_read(
                matrix, 'matrix'), 'matrix', i), 'matrix[i]', j)
            write_instrument_read(a, 'a')
            memory_module.malloc('a', sys.getsizeof(a))
            print(memory_module.locations['a'].location, 'a', 'mem')
else:
    print(1, 3)
