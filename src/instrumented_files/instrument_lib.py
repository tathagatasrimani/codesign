
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
        print(var_name, lower, upper, "Read")
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
        print(var_name, ind, "Read")
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
        print(var_name, lower, upper, "Write")
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
        print(var_name, ind, "Write")
        return var[ind]

def instrument_read(var, var_name: str):
    if not var_name:
        var_name = "None"
    print(var_name, "no_ind", "Read")
    return var

def write_instrument_read(var, var_name: str):
    if not var_name:
        var_name = "None"
    print(var_name, "no_ind", "Write")
    return var

class Object:
    def __init__(self, value):
        self.value = value