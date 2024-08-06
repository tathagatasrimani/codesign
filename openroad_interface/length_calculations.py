#[INFO GRT-0018] Total wirelength: 846 um

#[INFO GRT-0088] Layer metal1  Track-Pitch = 0.1400  line-2-Via Pitch: 0.1350
'''[INFO GRT-0088] Layer metal2  Track-Pitch = 0.1900  line-2-Via Pitch: 0.1400
[INFO GRT-0088] Layer metal3  Track-Pitch = 0.1400  line-2-Via Pitch: 0.1400
[INFO GRT-0088] Layer metal4  Track-Pitch = 0.2800  line-2-Via Pitch: 0.2800
[INFO GRT-0088] Layer metal5  Track-Pitch = 0.2800  line-2-Via Pitch: 0.2800
[INFO GRT-0088] Layer metal6  Track-Pitch = 0.2800  line-2-Via Pitch: 0.2800
[INFO GRT-0088] Layer metal7  Track-Pitch = 0.8000  line-2-Via Pitch: 0.8000
[INFO GRT-0088] Layer metal8  Track-Pitch = 0.8000  line-2-Via Pitch: 0.8000
[INFO GRT-0088] Layer metal9  Track-Pitch = 1.6000  line-2-Via Pitch: 1.6000
[INFO GRT-0088] Layer metal10 Track-Pitch = 1.6000  line-2-Via Pitch: 1.6000'''

import copy

def_file = "../test/results/final_generated.def"

### 0. reading def file ###
def_data = open(def_file)
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

# print(length_list )


