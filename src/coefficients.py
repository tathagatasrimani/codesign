import yaml

tech_params = yaml.load(open("tech_params.yaml", "r"), Loader=yaml.Loader)

latency = tech_params["latency"]
dynamic_power = tech_params["dynamic_power"]
leakage_power = tech_params["leakage_power"]

sizes = [3, 5, 7, 40]
coeffs = {
    "alpha": {},
    "beta": {},
    "gamma": {}
}

for elem in latency[3]:
    total = 0
    for size in sizes:
        total += latency[size][elem] / latency[size]["Invert"]
    coeffs["gamma"][elem] = total / len(sizes)

for elem in dynamic_power[3]:
    total = 0
    for size in sizes:
        total += dynamic_power[size][elem] / dynamic_power[size]["Invert"]
    coeffs["alpha"][elem] = total / len(sizes)

for elem in leakage_power[3]:
    total = 0
    for size in sizes:
        total += leakage_power[size][elem] / leakage_power[size]["Invert"]
    coeffs["beta"][elem] = total / len(sizes)
    
print(yaml.dump(coeffs))
