import copy
from var import directory


def length_calculations(units: float, def_file: str = "results/final_generated-tcl.def") -> dict: 
    '''
    calculates lengths of each net using the def file

    params: 
        units: to convert length into correct units
        def_file: the .def file contains macro coordinates, which are used for length calculations. 
        defaults to the name of the final generated .def file. 
    returns: 
        length_dict: length with macro ID attribution
    '''

    def_file = "results/final_generated-tcl.def"

    # parsing through def file for macro coords and net names
    def_data = open(directory + def_file)
    def_lines = def_data.readlines()
    length_dict = {}
    after_nets = False
    in_nets = False
    macro_ID = None
    for line in def_lines:
        if "END NETS" in line:
            after_nets = True
        if "NETS " in line and "SPECIAL" not in line:
            in_nets = True

        if in_nets == True and after_nets == False:
            if "- _" in line:
                data = line.split(" ")
                constant = copy.deepcopy(data)
                for item in constant:
                    if item == "":
                        data.remove("")
                macro_ID = data[1]
                length_dict[macro_ID] = 0
            elif "+ ROUTED " in line or "NEW metal" in line:
                line = line.split("(", 1)
                data = line[1].split(" ")
                constant = copy.deepcopy(data)
                curlength = 0 
                for item in constant:
                    if item == "":
                        data.remove("")
                    elif item == "(":
                        data.remove("(")
                    elif item == ")":
                        data.remove(")")
                    elif "via" in item:
                        data.remove(item)
                    elif "\n" in item:
                        data.remove(item)
                # doing length calculations here
                if len(data) > 2:
                    if data[2] == "*":
                        curlength = abs(int(data[1]) - int(data[3]))
                    elif data[3] == "*":
                        curlength = abs(int(data[0]) - int(data[2]))
                length_dict[macro_ID] += curlength/units
    return length_dict


