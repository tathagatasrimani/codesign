import numpy as np

from . import sim_util
from .global_constants import SYSTEM_BUS_SIZE

ALIGNMENT = 8
MAX_REQUEST_SIZE = 1 << 30

# This is a memory location allocator for variables inside of programs

rng = np.random.default_rng()


class block:
    def __init__(self, size, location, prev=None, nxt=None, dims=[], elem_size=0):
        self.size = size
        self.location = location
        self.prev = prev
        self.nxt = nxt
        self.dims = dims
        # maybe this elem_size shouldn't be a variable in this block class.
        # elem_size is variable specific, not memory specific
        self.elem_size = (
            elem_size if elem_size != 0 else size
        )  # used for arrays. size of individual element


class Memory:
    def __init__(self, segment_size: int, start_loc=0):
        self.locations = {}  # maps variable id to an allocated block
        self.segment_size: int = segment_size

        self.segment_start = block(segment_size, start_loc)

        # fblocks and ablocks are for debugging purposes
        # self.fblocks: int = 1
        # self.ablocks: int = 0
        self.first_free: block = self.segment_start
        self.last_free: block = self.segment_start

    # rounds up a request size to be correctly aligned
    def roundup(self, sz: int):
        return (sz + ALIGNMENT - 1) & ~(ALIGNMENT - 1)

    # Takes in two free blocks (presumably with one block in between them), and rewires them to
    # connect to each other rather than to the middle block. Updates first_free and last_free in
    # certain edge cases.
    def rewire_out(self, prev_free: block, next_free: block):
        if prev_free:
            prev_free.nxt = next_free
        else:
            self.first_free = next_free
        if next_free:
            next_free.prev = prev_free
        else:
            self.last_free = prev_free

    # Takes in three free blocks, with prev_free and next_free already wired to each other, and
    # changes the wiring so that all three are connected in order (prev, new, next). Updates first_free
    # and last_free in certain edge cases.
    def rewire_in(self, prev_free, new_free, next_free):
        new_free.prev = prev_free
        new_free.nxt = next_free
        if prev_free:
            prev_free.nxt = new_free
        else:
            self.first_free = new_free
        if next_free:
            next_free.prev = new_free
        else:
            self.last_free = new_free

    # used in the context of freeing a variable. Need to figure out where the other closest free block is
    # so we can add this block to the free list
    def orient_frees(self, cur_free):
        next_free = self.first_free
        while next_free and next_free.location < cur_free.location:
            next_free = next_free.nxt
        return next_free

    def parse_id(self, id):
        """
        How Currently this doesn't allow _ in actual variable names. treats _ as a priveleged character that only we 
        can use in the instrumentation.
        """
        words = id.split("_")
        return words[0]
        # if len(words) == 1:
        #     return id
        # else:
        #     if words[-1].isdigit():
        #         return "_".join(words[:-1])
        #     else:
        #         return id

    # Takes in a requested size, and clears out a space in the heap for that block, if possible.
    # Uses the first fit method, iterating through the free list from the start until the first suitable
    # block of free memory is found.
    def malloc(self, id, requested_size: int, dims, elem_size=0):
        id = self.parse_id(id)
        if id in self.locations:
            self.free(id)
        if requested_size == 0:
            return
        needed = self.roundup(requested_size)
        if needed > MAX_REQUEST_SIZE:
            return None
        free_block = self.first_free
        while free_block and free_block.size < needed:
            free_block = free_block.nxt
        if not free_block:
            return
        # split the free block if possible, and rewire as necessary
        prev_free = free_block.prev
        next_free = free_block.nxt
        alloc_block = None
        if free_block.size - needed < ALIGNMENT:
            # allocating the entire free block
            self.rewire_out(prev_free, next_free)
            # self.fblocks -= 1
            alloc_block = free_block
        else:
            # splitting the block and adding a new free one
            old_size = free_block.size
            free_block.size = needed
            new_free = block(
                size=old_size - needed, location=free_block.location + needed
            )
            self.rewire_in(prev_free, new_free, next_free)
            alloc_block = free_block
            alloc_block.dims = dims
        # self.ablocks += 1
        alloc_block.elem_size = elem_size
        self.locations[id] = alloc_block

    def free(self, id):
        id = self.parse_id(id)
        if id not in self.locations:
            return
        block_to_free = self.locations[id]
        self.locations.pop(id)
        next_free = self.orient_frees(block_to_free)
        prev_free = self.last_free
        if next_free:
            prev_free = next_free.prev
        self.rewire_in(prev_free, block_to_free, next_free)
        # self.fblocks += 1
        # self.ablocks -= 1

    def realloc(self, id, new_size):
        self.free(id)
        self.malloc(id, new_size)

    def read(self, id):
        """
        returns the size of the variable in memory
        variable names like `G_27` and 'G_13` and 'G' will be considered equivalent.
        These namings are due to the instrumented code.

        """
        id = self.parse_id(id)
        if id in self.locations:
            return self.locations[id].elem_size
        else:
            arr_name = sim_util.get_var_name_from_arr_access(id)
            if arr_name in self.locations:
                return self.locations[arr_name].elem_size
            return None


class Cache:
    def __init__(
        self,
        size,
        memory: Memory,
        var_size=1,
        line_size=SYSTEM_BUS_SIZE,
    ):
        """
        Caching not actually being used. Want to leave memory heirarchy as a
        later addition/ use existing mem simulators for it.

        adding var_size as a hack to only allow buffer of 1 variable, therefore forcing access
        to main memory on every read.
        """
        self.size = size
        self.line_size = line_size
        self.vars = {}  # dict of names to size
        self.var_size = var_size
        self.free_space = size
        self.used_space = 0
        self.memory = memory

    def find(self, var):
        if var in self.vars.keys():
            return True
        return False

    def evict(self, var, size):
        self.vars.pop(var)
        self.used_space -= size
        self.free_space += size

    def evict_random(self):
        var = rng.choice(list(self.vars.keys()))
        size = self.vars[var]
        self.evict(var, size)

    def read(self, var):
        """
        If cache hit, return true, if miss, update vars in cache and return false.
        returns positive size if it is a hit, negative size if it is a miss
        """
        if var in self.vars.keys():
            return self.vars[var]
        else:
            size = self.memory.read(var)
            if size is None:
                # TODO: I don't want to deal with this right now.
                # But there is issues with the instrumented code and variable names
                # that is resulting in this error.
                # for example, it considers math.inf as a variable name.
                # additionally, G_27 and G_26 were being considered as separate variables
                # I fixed that in the memory class, but it still doesn't account for math.inf
                # style issues.
                # raise Exception("variable not found in memory")
                size = 0
            if (
                len(self.vars) == self.var_size or size > self.free_space
            ):
                self.evict_random()
            self.vars[var] = size
            self.used_space += size
            self.free_space -= size
            return -1 * size

# storage unit keeps track of total space, used space, variables currently there, and trace of operations on it
class Buffer:
    def __init__(self, size):
        self.size = size
        self.vars = {}
        self.free_space = size
        self.ops = []
        self.last_write_op = {}
        self.dirty = {}

    def evict(self, var, size):
        self.vars.pop(var)
        self.free_space += size

    def evict_random(self):
        var = rng.choice(list(self.vars.keys()))
        size = self.vars[var]
        self.evict(var, size)
        self.ops.append(f"evict {var}")
        return var

    # write variable, return any which had to be evicted along with their last write ops
    def write(self, var, op_name, size=16):
        assert size <= self.size, f"Buffer does not have enough space for {var} of size {size}, opname {op_name}"
        evicted = []
        if var not in self.vars:
            while self.free_space - size < 0:
                evicted.append(self.evict_random())
            self.vars.add(var)
            self.free_space -= size
            self.ops.append(f"write {var} for {op_name}")
            self.dirty[var] = False
        else:
            self.ops.append(f"update value of {var} in {op_name}")
            self.dirty[var] = True
        self.last_write_op[var] = op_name
        retval = [(i, self.last_write_op[i], self.dirty[i]) for i in evicted]
        for evictee in evicted:
            self.last_write_op.pop(evictee)
            self.dirty.pop(evictee)
        return retval

    def check_hit(self, var):
        print(self.vars, var)
        if var in self.vars:
            self.ops.append(f"read {var}")
        return var in self.vars
    
class Register(Buffer):
    def __init__(self):
        super().__init__(16) # reg is 16 bit
    
    # register write is different because we assert that only 1 element should be there at a time
    def write(self, var, op_name, size=16):
        evicted = []
        if var not in self.vars:
            keys = list(self.vars.keys())
            if len(keys) > 0:
                evicted = keys
                assert len(evicted) == 1, "reg should hold 1 value"
                self.ops.append(f"evict {evicted[0]}")
            self.vars = {var: 16}
            self.ops.append(f"write {var}")
            self.dirty[var] = False
        else:
            self.ops.append(f"update value of {var}")
            self.dirty[var] = True
        self.last_write_op[var] = op_name
        retval = []
        if len(evicted) > 0:
            retval = [(evicted[0], self.last_write_op[evicted[0]], self.dirty[evicted[0]])]
            self.last_write_op.pop(evicted[0])
            self.dirty.pop(evicted[0])
        print(retval)
        return retval
        

# debugging
if __name__ == "__main__":
    test = Memory(1000)
    test.malloc("a", 128)
    print(test.locations["a"].location)
    test.malloc("b", 256)
    print(test.locations["b"].location)
    test.malloc("c", 512)
    print(test.locations["c"].location)
    test.free("b")
    test.malloc("d", 128)
    test.malloc("e", 64)
    print(test.locations["d"].location)
    print(test.locations["e"].location)
    print(test.locations)
    # print(test.fblocks, test.ablocks)
