import pandas as pd

# For now, only reading from asap7 data
# Will update in the future with more
f = open("tech_node_data/asap7data.txt", "r")
fl = f.readlines()
d = []
name = []
technode = []
for i in range(len(fl)):
    if fl[i].startswith("cell") and len(fl[i+2])>1:
        name.append(fl[i][fl[i].find("(")+1:fl[i].find("_")])
        technode.append(7)
        n = name[-1]
        d.append(["7", n, float(fl[i+1].split(" ")[1])*1e-6, float(fl[i+5].split(" ")[1])*1e-15, float(fl[i+6].split(" ")[1])])
ind = pd.MultiIndex.from_arrays([technode, name], names=("tech node", "standard cell"))
df = pd.DataFrame(data=d, index=ind, columns=["tech node", "standard cell", "area", "R_eff", "C_eff"])
df.to_csv("tech_node_data/cell_data.csv")