import yaml

V_dd = 1
f = 1e9

tech_params = yaml.load(open("tech_params.yaml", "r"), Loader=yaml.Loader)

latency, power = tech_params["latency"], tech_params["dynamic_power"]

sizes = [40]

rcs = {
    "Reff": {},
    "Ceff": {}
}

for elem in latency[3]:
    totalR = 0
    for size in sizes[-1:]:
        totalR += 0.5 * (latency[size][elem]/power[size][elem]) * V_dd * V_dd * f
    rcs["Reff"][elem] = totalR / len(sizes)
    rcs["Ceff"][elem] = latency[size][elem] / rcs["Reff"][elem]

with open("rcs.yaml", 'w') as f:
    f.write(yaml.dump(rcs))