import re 
        
def find_val_two(string, data, start):
    pattern_1 = fr'{string}\s+\S+\s+\S+\s;'
    pattern_2 = fr'{string}\s+\S+\sBY\s+\S+\s;'
    match = None
    result = []
    for num in range(len(data)):
        match_1 = re.search(pattern_1, data[start + num])
        match_2 = re.search(pattern_2, data[start + num])  
        if match_1 != None or match_2 != None:  
            match = match_1 or match_2
            break
    result.append(clean(value(match.group(0), string)).split(" ")[0])
    if " BY " in value(match.group(0), string):
        result.append(clean(value(match.group(0), string)).split(" ")[2])
    else:
        result.append(clean(value(match.group(0), string)).split(" ")[1])
    return result

def find_val_xy(string, data, start, x_or_y):
    result = find_val(string, data, start)
    if type(result) == list:
        if x_or_y == "x":
            return result[0]
        elif x_or_y == "y":
            return result[1]
    else:
        return result

def find_val(string, data, start):
    pattern = fr'{string}\s+\S+\s+;'
    match = None
    for num in range(len(data)):
        match = re.search(pattern, data[start + num])  
        if match != None:  
            break
        if string in data[start + num] and match == None and string != "WIDTH":
            return find_val_two(string, data, start)
    return clean(value(match.group(0), string))

def value(string, object):
    return str(string).split(object,1)[1]

def format(num):
    return "_" + (3 - len(str(num)))*"0" + str(num) + "_"

def clean(string):
    tmp = string.replace(";", "")
    if string.startswith(" "):
        tmp = tmp.replace(" ", "", 1)
    tmp = tmp.replace("\n", "")
    if tmp.endswith(" "):
        tmp = tmp[:-1]
    return tmp
