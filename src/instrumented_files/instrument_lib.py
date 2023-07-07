
def instrument_read_sub(var, var_name: str, ind):
    print(var_name, ind, "Read")
    return var[ind]

def write_instrument_read_sub(var, var_name: str, ind):
    print(var_name, ind, "Write")
    return var[ind]

def instrument_read(var, var_name: str):
    print(var_name, "no_ind", "Read")
    return var

def write_instrument_read(var, var_name: str):
    print(var_name, "no_ind", "Write")
    return var