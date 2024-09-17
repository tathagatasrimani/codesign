import re

with open(test file, 'r') as file:
    lines = file.readlines()

# Modify specific lines
for line in lines:
    if "set die_area" in line:
        coordinates = "{{} {} {} {}}".format(1,1,1,1)
        line = re.sub(r'\{.*?\}', coordinates, line)
    if "set core_area" in line:
        coordinates = "{{} {} {} {}}".format(1,1,1,1)
        line = re.sub(r'\{.*?\}', coordinates, line)

# Write the modified lines back to the file
with open('example.txt', 'w') as file:
    file.writelines(lines)