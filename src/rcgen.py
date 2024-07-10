import yaml
import cacti.cacti_python.get_dat as dat
import math

# Read in dat file, add to the rcs file
def generate_optimization_params(latency, active_power, active_energy, passive_power, V_dd, dat_file):
    """
    Generate R,C, etc from the latency, power tech parameters.
    rcs[other] are all stored in SI units.
        V_dd: voltage in V
        MemReadL: memory read latency in s
        MemWriteL: memory write latency in s
        MemReadEact: memory read active power in W
        MemWriteEact: memory write active power in W
        MemPpass: memory passive power in W
    
        params:
        latency: dictionary of latencies in cycles
        active_power: dictionary of active power in nW
        passive_power: dictionary of passive power in nW
        V_dd: voltage in V
    """
    rcs = {"Reff": {}, "Ceff": {}, "Cacti": {}, "other": {}}

    rcs["other"]["V_dd"] = V_dd

    # store in ns
    rcs["other"]["MemReadL"] = latency["MainMem"]
    rcs["other"]["MemWriteL"] = latency["MainMem"]

    # store in nW and nJ
    rcs["other"]["MemReadEact"] = active_energy["MainMem"]["Read"] * 1e-9
    rcs["other"]["MemWriteEact"] = active_energy["MainMem"]["Write"] * 1e-9
    rcs["other"]["MemPpass"] = passive_power["MainMem"] * 1e-9

    rcs["other"]["BufL"] = latency["Buf"] # ns
    
    # store in nW and nJ
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

    # CACTI
    cacti_params = {}
    # TODO, cell type, temp
    dat.scan_dat(cacti_params, dat_file, 0, 0, 360)
    cacti_params = {k: (1 if v is None or math.isnan(v) else (10**(-9) if v == 0 else v)) for k, v in cacti_params.items()}
    for key, value in cacti_params.items():
        rcs["Cacti"][key] = value

    return rcs


def main():
    V_dd = 1.1

    tech_params = yaml.load(open("params/tech_params.yaml", "r"), Loader=yaml.Loader)

    size = 7

    latency = tech_params["latency"][size]
    active_power = tech_params["dynamic_power"][size]
    active_energy = tech_params["dynamic_energy"][size]
    passive_power = tech_params["leakage_power"][size]

    rcs = generate_optimization_params(latency, active_power, active_energy, passive_power, V_dd)

    with open("params/rcs.yaml", 'w') as f:
        f.write(yaml.dump(rcs))

if __name__ == "__main__":
    main()