import sys


def instrument_read_sub(var, var_name: str, ind, lower, upper, slice, f):
    while var_name.startswith("instrument_read"):
        var_name = var_name[var_name.find('(')+1:]
    if var_name.find(',') != -1:
        var_name = var_name[:var_name.find(',')]
    if var_name.find('[') != -1:
        var_name = var_name[:var_name.find('[')]
    if var_name == "":
        var_name = "None"
    if slice:
        f.write(str(var_name) + ' ' + str(lower) + ' ' + str(upper) + " Read " + str(sys.getsizeof(var[lower:upper])) + '\n')
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
        f.write(str(var_name) + ' ' + str(ind) + " Read " + str(sys.getsizeof(var[ind])) + '\n')
        return var[ind]

def write_instrument_read_sub(var, var_name: str, ind, lower, upper, slice, f):
    while var_name.startswith("instrument_read"):
        var_name = var_name[var_name.find('(')+1:]
    if var_name.find(',') != -1:
        var_name = var_name[:var_name.find(',')]
    if var_name.find('[') != -1:
        var_name = var_name[:var_name.find('[')]
    if var_name == "":
        var_name = "None"
    if slice:
        f.write(str(var_name) + ' ' + str(lower) + ' ' + str(upper) + " Write " + str(sys.getsizeof(var[lower:upper])) + '\n')
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
        f.write(str(var_name) + ' ' + str(ind) + " Write " + str(sys.getsizeof(var[ind])) + '\n')
        return var[ind]

def instrument_read(var, var_name: str, f):
    if not var_name:
        var_name = "None"
    f.write(str(var_name) + " no_ind Read " + str(sys.getsizeof(var)) + '\n')
    return var

def write_instrument_read(var, var_name: str, f):
    if not var_name:
        var_name = "None"
    f.write(str(var_name) + " no_ind Write " + str(sys.getsizeof(var)) + '\n')
    return var

def instrument_read_from_file(func, f, *args):
    vals = func(*args)
    f.write(str(func.__name__) + " no_ind Read NVM " + str(sys.getsizeof(vals)) + '\n')
    return vals

class Object:
    def __init__(self, value):
        self.value = value