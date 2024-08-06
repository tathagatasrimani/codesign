import re 

def find_val(string, data, start ):
    pattern = fr'{string}\s+\S+\s+;'
    match = None
    for num in range(len(data)):
        match = re.search(pattern, data[start + num])  
        if match != None:  
            break
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
