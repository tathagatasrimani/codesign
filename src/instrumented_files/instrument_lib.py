import sys
import numpy as np

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
        if type(var[ind]) is np.ndarray:
            print(f"{var_name} {var[ind].tolist()} Read {sys.getsizeof(var[ind])}")
        else:
            print(f"{var_name} {var[ind]} Read {sys.getsizeof(var[ind])}")
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
        if type(var[ind]) is np.ndarray:
            print(f"{var_name} {var[ind].tolist()} Write {sys.getsizeof(var[ind])}")
        else:
            print(f"{var_name} {var[ind]} Write {sys.getsizeof(var[ind])}")
        return var[ind]

def instrument_read(var, var_name: str):
    if not var_name:
        var_name = "None"
    if type(var) is np.ndarray:
        val = var.tolist() #np.array2string(var, max_line_width=np.inf)
    else:
        val = var
    print(
        f"{var_name} {val} Read {sys.getsizeof(var)}"
    )
    return var

def write_instrument_read(var, var_name: str):
    if not var_name:
        var_name = "None"
    if type(var) is np.ndarray:
        val = var.tolist() #np.array2string(var, max_line_width=np.inf)
    else:
        val = var
    print(
        f"{var_name} {val} Write {sys.getsizeof(var)}"
    )
    return var

def instrument_read_from_file(func, *args):
    vals = func(*args)
    print(
        f"{func.__name__} {vals.tolist()} Read NVM {sys.getsizeof(vals)}"
    )
    return vals

class Object:
    def __init__(self, value):
        self.value = value
