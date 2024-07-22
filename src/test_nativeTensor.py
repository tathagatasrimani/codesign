class nativeTensor():
    def __init__(self, data):
        self.data = data
        self.shape = self._get_shape_recursive(data)
        ''' Only support broadcasting one dimension'''
        self.broadcast_dim = None
        self.broadcast_length = None
    
    def _get_shape_recursive(self, data):
        if isinstance(data, list):
            return [len(data)] + self._get_shape_recursive(data[0])
        return []
    
    def set_broadcast(self, dim, length):
        self.broadcast_dim = dim
        self.broadcast_length = length
        self.shape.insert(dim, length)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        data = self.data
        if self.broadcast_dim != None:
            if len(indices) <= self.broadcast_dim:
                pass
            elif indices[self.broadcast_dim] >= self.broadcast_length:
                raise IndexError("Index out of range")
            else:
                list_indices = list(indices)
                list_indices.pop(self.broadcast_dim)
                indices = tuple(list_indices)
        for idx in indices:
            data = data[idx]
        return nativeTensor(data) if isinstance(data, list) else data
    
    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]
        
a = nativeTensor([1,2])
print(f"a[0] is {a[0]}")
print(f"a has shape {a.shape}")
for i in a:
    print(i)

a.set_broadcast(0,8)
print(f"after broadcast, a[4,0] is {a[4,0]}")
print(f"after broadcast, a has shape {a.shape}")
for i in a:
    print("")
    for j in i:
        print(j, end="")
