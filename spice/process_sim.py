import matplotlib.pyplot as plt
import numpy as np

def parse_spectre_raw(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find start of "Values:" section
    for i, line in enumerate(lines):
        if line.strip().startswith("Values:"):
            data_start = i + 1
            break
    else:
        raise ValueError("No 'Values:' section found in raw file.")

    # Parse variable headers
    headers = ["time"]
    var_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Variables:"):
            var_index = i
            break
    else:
        raise ValueError("No 'Variables:' section found in raw file.")
    
    headers = [lines[var_index].strip().split()[2]]
    for i in range(var_index+1, data_start-1):
        headers.append(lines[i].strip().split()[1])
    
    print(headers)

    # Load data
    data = []
    ind = 0
    for line in lines[data_start:]:
        if line.strip():
            if line.strip().split()[0] == str(ind):
                row = [float(x) for x in line.strip().split()]
                data.append(row)
                ind += 1
            else:
                row = [float(x) for x in line.strip().split()]
                data[-1].extend(row)

    data = np.array(data)
    print(data)
    print(headers)
    return data[:,1], {headers[i]: data[:,i+1] for i in range(1, len(headers))}

# --- Usage ---
time, signals = parse_spectre_raw('inverter_test.raw')

print(signals)

# Plot
"""plt.plot(time, signals['in'], label='Vin')
plt.plot(time, signals['out'], label='Vout')
plt.plot(time, signals['vdd'], label='Vdd')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Spectre Transient Simulation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()"""

def find_transitions(signal):
    i = 0
    transitions = []
    while(i < len(signal)):
        if signal[i] > np.max(signal) / 2:
            vals = np.where(signal[i:] < np.max(signal) / 2)[0]
        else:
            vals = np.where(signal[i:] > np.max(signal) / 2)[0]
        if len(vals) == 0:
            break
        i += vals[0]
        print(i)
        
        transitions.append(i)
    
    return transitions

print(signals['in'][20:])
print(signals['out'][20:])
# don't want to include the initial transient
in_half_vdd = find_transitions(signals['in']) 
out_half_vdd = find_transitions(signals['out']) 

print(in_half_vdd, out_half_vdd)

print([signals['in'][ind] for ind in in_half_vdd])
print([signals['out'][ind] for ind in out_half_vdd])
print(signals['out'][0], signals['out'][len(signals['out'])//4])
print("tpd: ", time[out_half_vdd[1]] - time[in_half_vdd[1]])

