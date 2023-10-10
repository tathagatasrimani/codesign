import sys


def instrument_read_sub(var, var_name: str, ind, lower, upper, slice):
    while var_name.startswith("instrument_read"):
        var_name = var_name[var_name.find('(')+1:]
    if var_name.find(',') != -1:
        var_name = var_name[:var_name.find(',')]
    if var_name.find('[') != -1:
        var_name = var_name[:var_name.find('[')]
    if var_name == "":
        var_name = "None"
    if slice:
        print(var_name, lower, upper, "Read", sys.getsizeof(var[lower:upper]))
        if lower:
            if upper:
                return var[int(lower):int(upper)]
            else:
                return var[int(lower):]
        else:
            if upper:
                return var[:int(upper)]
            else:
                return var
    else:
        print(var_name, ind, "Read", sys.getsizeof(var[ind]))
        return var[ind]

def write_instrument_read_sub(var, var_name: str, ind, lower, upper, slice):
    while var_name.startswith("instrument_read"):
        var_name = var_name[var_name.find('(')+1:]
    if var_name.find(',') != -1:
        var_name = var_name[:var_name.find(',')]
    if var_name.find('[') != -1:
        var_name = var_name[:var_name.find('[')]
    if var_name == "":
        var_name = "None"
    if slice:
        print(var_name, lower, upper, "Write", sys.getsizeof(var[lower:upper]))
        if lower:
            if upper:
                return var[int(lower):int(upper)]
            else:
                return var[int(lower):]
        else:
            if upper:
                return var[:int(upper)]
            else:
                return var
    else:
        print(var_name, ind, "Write", sys.getsizeof(var[lower:upper]))
        return var[ind]

def instrument_read(var, var_name: str):
    if not var_name:
        var_name = "None"
    print(var_name, "no_ind", "Read", sys.getsizeof(var))
    return var

def write_instrument_read(var, var_name: str):
    if not var_name:
        var_name = "None"
    print(var_name, "no_ind", "Write", sys.getsizeof(var))
    return var

# def instrument_read_from_file(func):


class Object:
    def __init__(self, value):
        self.value = value