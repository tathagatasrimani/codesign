from schedule import schedule
from hls import HardwareModel
from ast_utils import ASTUtils
import matplotlib.pyplot as plt
import ast
import hls
import math
import json
import sys

power = {
    "And": 32 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Or": 32 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Add": [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Sub": [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Mult": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "FloorDiv": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "Mod": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "LShift": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "RShift": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "BitOr": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitXor": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitAnd": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Eq": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "NotEq": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Lt": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "LtE": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Gt": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "GtE": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "USub": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "UAdd": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "IsNot": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Not": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Invert": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Regs": [7.936518e-03, 1.062977e-03, 8.999495e-03, 8.999495e-03, 7.395312e-05],
}

data = {}
cycles = 0
main_cfg = None
id_to_node = {}
path = '/Users/PatrickMcEwen/git_container/codesign/src/cfg/benchmarks/' # change path variable for local computer
benchmark = 'simple'
data_path = []
power_use = []
node_intervals = []
node_avg_power = {}

def func_calls(expr, calls):
    if type(expr) == ast.Call:
        calls.append(expr.func.id)
    for sub_expr in ASTUtils.get_sub_expr(expr):
        func_calls(sub_expr, calls)

def get_hw_need(state):
    hw_need = HardwareModel(0,0)
    for op in state:
        if op.operation == 'Read' or op.operation == 'Write': hw_need.hw_allocated['Regs'] += 1
        else: hw_need.hw_allocated[op.operation] += 1
    return hw_need.hw_allocated

def cycle_sim(hw_inuse, max_cycles):
    global cycles, data, power_use
    start_cycles = cycles
    node_power_sum = 0
    for i in range(max_cycles):
        power_use.append(0)
        #print("This is during cycle " + str(cycles))
        #print(hw_inuse)
        #print("")
        # save current state of hardware to data array
        cur_data = ""
        for elem in hw_inuse:
            if len(hw_inuse[elem]) > 0:
                cur_data += elem + ": "
                count = 0
                for i in hw_inuse[elem]:
                    if i > 0:
                        count += 1
                        power_use[cycles] += power[elem][2]
                power_use[cycles] += (power[elem][2] / 10) * len(hw_inuse[elem]) # passive power
                cur_data += str(count) + "/" + str(len(hw_inuse[elem])) + " in use. || "
        data[cycles] = cur_data
        # simulate one cycle
        for elem in hw_inuse:
            for j in range(len(hw_inuse[elem])):
                if hw_inuse[elem][j] > 0:
                    hw_inuse[elem][j] = max(0, hw_inuse[elem][j] - 1)
        node_power_sum += power_use[cycles]
        cycles += 1
    return node_power_sum

def simulate(cfg, data_path, node_operations, hw_spec, first):
    global main_cfg, id_to_node
    cur_node = cfg.entryblock
    if first: 
        cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        main_cfg = cfg
    hw_inuse = {}
    for elem in hw_spec:
        hw_inuse[elem] = [0] * hw_spec[elem]
    #print(hw_inuse)

    for elem in data_path:
        node_id = elem[0]
        cur_node = id_to_node[node_id]
        node_intervals.append([node_id, [cycles, 0]])
        node_avg_power[node_id] = 0 # just reset because we will end up overwriting it
        start_cycles = cycles # for calculating average power
        for state in node_operations[cur_node]:
            hw_need = get_hw_need(state)
            #print(hw_need)
            max_cycles = 0
            for elem in hw_need:
                cur_elem_count = hw_need[elem]
                if cur_elem_count == 0: continue
                if hw_spec[elem] == 0 and cur_elem_count > 0:
                    raise Exception("hardware specification insufficient to run program")
                cur_cycles_needed = int(math.ceil(cur_elem_count / hw_spec[elem]) * hls.latency[elem])
                #print("cycles needed for " + elem + ": " + str(cur_cycles_needed) + ' (element count = ' + str(cur_elem_count) + ')')
                max_cycles = max(cur_cycles_needed, max_cycles)
                i = 0
                while cur_elem_count > 0:
                    hw_inuse[elem][i] += hls.latency[elem]
                    i = (i + 1) % hw_spec[elem]
                    cur_elem_count -= 1
            node_avg_power[node_id] += cycle_sim(hw_inuse, max_cycles)
        if cycles - start_cycles > 0: node_avg_power[node_id] /= cycles - start_cycles
        node_intervals[-1][1][1] = cycles
    return data

def main():
    global power_use
    if len(sys.argv) > 1:
        benchmark = sys.argv[1]
    cfg, graphs, node_operations = schedule(sys.argv[1])
    hw = HardwareModel(0, 0)
    hw.hw_allocated['Add'] = 1
    hw.hw_allocated['Regs'] = 3
    hw.hw_allocated['Mult'] = 1
    hw.hw_allocated['Sub'] = 1
    hw.hw_allocated['FloorDiv'] = 1
    hw.hw_allocated['Gt'] = 1
    hw.hw_allocated['And'] = 1
    hw.hw_allocated['Or'] = 1
    hw.hw_allocated['Mod'] = 1
    hw.hw_allocated['LShift'] = 1
    hw.hw_allocated['RShift'] = 1
    hw.hw_allocated['BitOr'] = 1
    hw.hw_allocated['BitXor'] = 1
    hw.hw_allocated['BitAnd'] = 1
    hw.hw_allocated['Eq'] = 1
    hw.hw_allocated['NotEq'] = 1
    hw.hw_allocated['Lt'] = 1
    hw.hw_allocated['LtE'] = 1
    hw.hw_allocated['Gt'] = 1
    hw.hw_allocated['GtE'] = 1
    for node in cfg:
        id_to_node[str(node.id)] = node
    # set up sequence of cfg nodes to visit
    with open('/Users/PatrickMcEwen/git_container/codesign/src/instrumented_files/output.txt', 'r') as f:
        src = f.read()
        l = src.split('\n')
        for i in range(len(l)):
            l[i] = l[i].split()
        #print(l)
        last_line = '-1'
        last_node = '-1'
        for item in l:
            if len(item) == 2 and (item[0] != last_node or item[1] == last_line):
                last_node = item[0]
                last_line = item[1]
                data_path.append(item)      
    data = simulate(cfg, data_path, node_operations, hw.hw_allocated, first=True)
    text = json.dumps(data, indent=4)
    names = sys.argv[1].split('/')
    with open(path + 'json_data/' + names[-1], 'w') as fh:
        fh.write(text)
    t = []
    for i in range(len(power_use)):
        t.append(i)
    #print(power_use)
    plt.plot(t,power_use)
    plt.title("power use for " + names[-1])
    plt.xlabel("Cycle")
    plt.ylabel("Power")
    plt.savefig("cfg/benchmarks/power_plots/power_use_" + names[-1] + ".pdf")
    plt.clf() 

if __name__ == '__main__':
    main()