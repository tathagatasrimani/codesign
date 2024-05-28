import yaml

def generate_optimization_params(latency, active_power, active_energy, passive_power, V_dd):
    """
    Generate R,C, etc from the latency, power tech parameters.
    rcs[other] are all stored in SI units.
        V_dd: voltage in V
        MemReadL: memory read latency in s
        MemWriteL: memory write latency in s
        MemReadPact: memory read active power in W
        MemWritePact: memory write active power in W
        MemPpass: memory passive power in W
    
        params:
        latency: dictionary of latencies in cycles
        active_power: dictionary of active power in nW
        passive_power: dictionary of passive power in nW
        V_dd: voltage in V
    """
    rcs = {"Reff": {}, "Ceff": {}, "other": {}}

    rcs["other"]["V_dd"] = V_dd

    rcs["other"]["MemReadL"] = latency["MainMem"]
    rcs["other"]["MemWriteL"] = latency["MainMem"]
    rcs["other"]["MemReadEact"] = active_energy["MainMem"] * 1e-9
    rcs["other"]["MemWriteEact"] = active_energy["MainMem"] * 1e-9
    rcs["other"]["MemPpass"] = passive_power["MainMem"] * 1e-9

    rcs["other"]["BufL"] = latency["Buf"]
    rcs["other"]["BufReadEact"] = active_energy["Buf"]["Read"] * 1e-9
    rcs["other"]["BufWriteEact"] = active_energy["Buf"]["Write"] * 1e-9
    rcs["other"]["BufPpass"] = passive_power["Buf"] * 1e-9

    rcs["other"]["OffChipIOL"] = latency["OffChipIO"]
    rcs["other"]["OffChipIOPact"] = active_power["OffChipIO"] * 1e-3 # this is in mW

    for elem in latency:
        if elem in ["Buf", "MainMem", "OffChipIO"]:
            continue
        R = 0.5 * V_dd**2 / (active_power[elem] * 1e-9)
        rcs["Reff"][elem] = R
        rcs["Ceff"][elem] = latency[elem] / rcs["Reff"][elem] # latency is ns so C is nF

    inv_R_off = V_dd**2 / (passive_power["Not"] * 1e-9)

    rcs["other"]["Roff_on_ratio"] = inv_R_off / rcs["Reff"]["Not"]

    return rcs


def main():
    V_dd = 1.1

    tech_params = yaml.load(open("tech_params.yaml", "r"), Loader=yaml.Loader)

    size = 7

    latency = tech_params["latency"][size]
    active_power = tech_params["dynamic_power"][size]
    active_energy = tech_params["dynamic_energy"][size]
    passive_power = tech_params["leakage_power"][size]

    rcs = generate_optimization_params(latency, active_power, active_energy, passive_power, V_dd)

    with open("rcs.yaml", 'w') as f:
        f.write(yaml.dump(rcs))

if __name__ == "__main__":
    main()