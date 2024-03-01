import yaml

f_measurement = 5e9

def generate_optimization_params(latency, active_power, passive_power, V_dd, f):
    """
    Generate R,C, etc from the latency, power tech parameters.
    """
    rcs = {"Reff": {}, "Ceff": {}, "other": {}}

    rcs["other"]["f"] = f
    rcs["other"]["V_dd"] = V_dd

    rcs["other"]["MemReadL"] = latency["MainMem"] / f_measurement
    rcs["other"]["MemWriteL"] = latency["MainMem"] / f_measurement
    rcs["other"]["MemReadPact"] = active_power["MainMem"] * 1e-9
    rcs["other"]["MemWritePact"] = active_power["MainMem"] * 1e-9
    rcs["other"]["MemPpass"] = passive_power["MainMem"] * 1e-9

    for elem in latency:
        if elem in ["Buf", "MainMem"]:
            continue
        R = 0.5 * ((latency[elem] / f_measurement) / (active_power[elem] * 1e-9)) * V_dd**2 * f
        rcs["Reff"][elem] = R
        rcs["Ceff"][elem] = (latency[elem] / f_measurement) / rcs["Reff"][elem]

    inv_R_off = V_dd**2 / (passive_power["Not"] * 1e-9)

    rcs["other"]["Roff_on_ratio"] = inv_R_off / rcs["Reff"]["Not"]

    return rcs


def main():
    V_dd = 1.1
    f = 2e9

    tech_params = yaml.load(open("tech_params.yaml", "r"), Loader=yaml.Loader)

    size = 40

    latency = tech_params["latency"][size]
    active_power = tech_params["dynamic_power"][size]
    passive_power = tech_params["leakage_power"][size]

    rcs = generate_optimization_params(latency, active_power, passive_power, V_dd, f)

    with open("rcs.yaml", 'w') as f:
        f.write(yaml.dump(rcs))

if __name__ == "__main__":
    main()