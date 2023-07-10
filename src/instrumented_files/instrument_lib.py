
def instrument_read_sub(var, var_name: str, ind):
    print(var_name, id(var[ind]), ind, "Read")
    return var[ind]

def write_instrument_read_sub(var, var_name: str, ind):
    print(var_name, id(var[ind]), ind, "Write")
    return var[ind]

def instrument_read(var, var_name: str):
    print(var_name, id(var), "no_ind", "Read")
    return var

def write_instrument_read(var, var_name: str):
    print(var_name, id(var), "no_ind", "Write")
    return var

class Object:
    def __init__(self, value):
        self.value = value