def length_calculations(): 
    import copy
    from var import directory

    def_file = "results/final_generated-tcl.def"

    ### 0. reading def file ###
    def_data = open(directory + def_file)
    def_lines = def_data.readlines()
    length_list = {}
    after_nets = False
    in_nets = False
    macro_ID = None


    # did info parsing here because i needed the coordinates and the def_generator doesnt have coords 
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
                length_list[macro_ID] = 0
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
                if len(data) > 2:
                    if data[2] == "*":
                        curlength = abs(int(data[1]) - int(data[3]))
                    elif data[3] == "*":
                        curlength = abs(int(data[0]) - int(data[2]))
                length_list[macro_ID] += curlength/2000
    return length_list
# print(length_list )


