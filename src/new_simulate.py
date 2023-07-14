from hardwareModel import HardwareModel
from ast_utils import ASTUtils
from memory import Memory
import schedule
import dfg_algo
import matplotlib.pyplot as plt
import ast
import hardwareModel
import math
import json
import sys
from collections import deque

MEMORY_SIZE = 1000000
memory_module = Memory(MEMORY_SIZE)

data = {}
cycles = 0
main_cfg = None
id_to_node = {}
path = '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/' # change path variable for local computer
data_path = []
power_use = []
node_intervals = []
node_avg_power = {}
unroll_at = {}
vars_allocated = {}
where_to_free = {}
memory_needed = 0
cur_memory_size = 0

def find_free_loc(var_name, split_lines, ind):
    where_to_free[var_name] = ind+1
    for i in range(ind+1, len(split_lines)):
        item = split_lines[i]
        if len(item) == 0: continue
        if item[0] == var_name: # can add condition item[2] == "Read" for extra optimization, choosing not to for now
            if item[-1] == "Read":
                where_to_free[var_name] = i+1
            else:
                where_to_free[var_name] = i

def func_calls(expr, calls):
    if type(expr) == ast.Call:
        calls.append(expr.func.id)
    for sub_expr in ASTUtils.get_sub_expr(expr):
        func_calls(sub_expr, calls)

def get_hw_need(state):
    hw_need = HardwareModel(0,0)
    for op in state:
        if not op.operation: continue
        else: hw_need.hw_allocated[op.operation] += 1
    return hw_need.hw_allocated

def cycle_sim(hw_inuse, max_cycles):
    global cycles, data, power_use
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
                        power_use[cycles] += hardwareModel.power[elem][2]
                power_use[cycles] += (hardwareModel.power[elem][2] / 10) * len(hw_inuse[elem]) # passive power
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

def process_compute_element(op, graph, node_id):
    global memory_module
    cfg_node = id_to_node[node_id]
    for parent in op.parents:
        if not parent.operation: break
        if parent.operation == "Regs":
            name = parent.value
            bracket_ind = name.find('[')
            if bracket_ind != -1:
                name = name[:bracket_ind]
            if name in memory_module.locations:
                mem_loc = memory_module.locations[name]
                text = name + "\nlocation: " + str(mem_loc.location)
                dfg_node_id = dfg_algo.set_id()
                print(dfg_node_id)
                dfg_algo.make_node(graph.gv_graph, cfg_node, dfg_node_id, text, ast.Load, parent.operation)
                print("node made for ", name)
            

def process_memory_operation(var_name, size, status):
    global memory_module
    if status == "malloc":
        memory_module.malloc(var_name[:var_name.rfind('_')], size)
    elif status == "free":
        memory_module.free(var_name[:var_name.rfind('_')], size)

# adds all mallocs and frees to vectors, and finds the next cfg node in the data path,
# returning the index of that node
def find_next_data_path_index(i, mallocs, frees, data_path):
    while len(data_path[i]) > 2:
        if data_path[i][0] == "malloc": mallocs.append(data_path[i])
        else: frees.append(data_path[i])
        i += 1
        if i == len(data_path): break
    return i

def simulate(cfg, node_operations, hw_spec, graphs, first):
    global main_cfg, id_to_node, unroll_at, memory_module, data_path
    cur_node = cfg.entryblock
    if first: 
        cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        main_cfg = cfg
    hw_inuse = {}
    for elem in hw_spec:
        hw_inuse[elem] = [0] * hw_spec[elem]
    #print(hw_inuse)
    i = 0
    frees = []
    mallocs = []
    print(data_path)
    i = find_next_data_path_index(i, mallocs, frees, data_path)
    while i < len(data_path):
        next_ind = find_next_data_path_index(i+1, mallocs, frees, data_path)
        if i == len(data_path): break
        for malloc in mallocs:
            process_memory_operation(malloc[2], int(malloc[1]), malloc[0])
        node_id = data_path[i][0]
        print(node_id, memory_module.locations)
        print(i)
        cur_node = id_to_node[node_id]
        node_intervals.append([node_id, [cycles, 0]])
        node_avg_power[node_id] = 0 # just reset because we will end up overwriting it
        start_cycles = cycles # for calculating average power
        iters = 0
        if unroll_at[cur_node.id]:
            j = next_ind
            while True:
                if len(data_path) <= j: break
                next_node_id = data_path[j][0]
                if next_node_id != node_id: break
                iters += 1
                j = find_next_data_path_index(j+1, [], [], data_path)
            next_ind = j
        for state in node_operations[cur_node]:
            # if unroll, take each operation in a state and create more of them
            if unroll_at[cur_node.id]:
                new_state = state.copy()
                for op in state:
                    for j in range(iters):
                        new_state.append(op)
                state = new_state
            hw_need = get_hw_need(state)
            for op in state:
                if op.operation and op.operation != "Regs":
                    process_compute_element(op, graphs[id_to_node[node_id]], node_id)
            #print(hw_need)
            max_cycles = 0
            for elem in hw_need:
                cur_elem_count = hw_need[elem]
                if cur_elem_count == 0: continue
                if hw_spec[elem] == 0 and cur_elem_count > 0:
                    raise Exception("hardware specification insufficient to run program")
                cur_cycles_needed = int(math.ceil(cur_elem_count / hw_spec[elem]) * hardwareModel.latency[elem])
                #print("cycles needed for " + elem + ": " + str(cur_cycles_needed) + ' (element count = ' + str(cur_elem_count) + ')')
                max_cycles = max(cur_cycles_needed, max_cycles)
                j = 0
                while cur_elem_count > 0:
                    hw_inuse[elem][j] += hardwareModel.latency[elem]
                    j = (j + 1) % hw_spec[elem]
                    cur_elem_count -= 1
            node_avg_power[node_id] += cycle_sim(hw_inuse, max_cycles)
        if cycles - start_cycles > 0: node_avg_power[node_id] /= cycles - start_cycles
        node_intervals[-1][1][1] = cycles
        for free in frees:
            if free[2] in memory_module.locations: 
                process_memory_operation(free[2], int(free[1]), free[0])
        mallocs.clear()
        frees.clear()
        graphs[id_to_node[node_id]].gv_graph.render(path + 'pictures/' + "test" + "_dfg_node_" + str(node_id), view = True)
        i = next_ind
    print("done with simulation")
    return data

def set_data_path():
    global data_path, cur_memory_size, vars_allocated, where_to_free, memory_needed
    with open('/Users/PatrickMcEwen/git_container/codesign/src/instrumented_files/output.txt', 'r') as f:
        f_new = open('/Users/PatrickMcEwen/git_container/codesign/src/instrumented_files/output_free.txt', 'w+')
        src = f.read()
        l = src.split('\n')
        split_lines = []
        for i in range(len(l)):
            split_lines.append(l[i].split())
        #print(l)
        last_line = '-1'
        last_node = '-1'
        for i in range(len(split_lines)):
            item = split_lines[i]
            f_new.write(l[i] + '\n')
            vars_to_pop = []
            for var_name in where_to_free:
                if where_to_free[var_name] == i:
                    var_size = vars_allocated[var_name]
                    f_new.write("free " + str(var_size) + " " + var_name + "\n")
                    data_path.append(["free", str(var_size), var_name])
                    vars_to_pop.append(var_name)
                    cur_memory_size -= var_size
                    vars_allocated.pop(var_name)
            for var_name in vars_to_pop:
                where_to_free.pop(var_name)
            if len(item) == 2 and (item[0] != last_node or item[1] == last_line):
                last_node = item[0]
                last_line = item[1]
                data_path.append(item)      
            elif len(item) == 3 and item[0] == "malloc" and item[2] not in vars_allocated:
                data_path.append(item)
                vars_allocated[item[2]] = int(item[1])
                print(vars_allocated)
                if item[2] not in where_to_free:
                    find_free_loc(item[2], split_lines, i)
                cur_memory_size += int(item[1])
                memory_needed = max(memory_needed, cur_memory_size)
    print(data_path)
    print("memory needed: ", memory_needed)

def simulator_prep(benchmark):
    global unroll_at, id_to_node
    cfg, graphs, unroll_at = dfg_algo.main_fn(path, benchmark)
    node_operations = schedule.schedule(cfg, graphs, benchmark)
    set_data_path()
    for node in cfg: id_to_node[str(node.id)] = node
    return cfg, graphs, node_operations

def main():
    global power_use
    benchmark = sys.argv[1]
    print(benchmark)
    cfg, graphs, node_operations = simulator_prep(benchmark)
    hw = HardwareModel(0, 0)
    hw.hw_allocated['Add'] = 15
    hw.hw_allocated['Regs'] = 30
    hw.hw_allocated['Mult'] = 15
    hw.hw_allocated['Sub'] = 15
    hw.hw_allocated['FloorDiv'] = 15
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
    hw.hw_allocated['GtE'] = 1
    hw.hw_allocated['IsNot'] = 1
    hw.hw_allocated['USub'] = 1
    hw.hw_allocated['UAdd'] = 1
    hw.hw_allocated['Not'] = 1
    hw.hw_allocated['Invert'] = 1
    
    data = simulate(cfg, node_operations, hw.hw_allocated, graphs, True)
    names = sys.argv[1].split('/')
    if len(sys.argv) < 3 or not sys.argv[2] == "notrace":
        text = json.dumps(data, indent=4)
        with open(path + 'json_data/' + names[-1], 'w') as fh:
            fh.write(text)
    t = []
    for i in range(len(power_use)):
        t.append(i)
    plt.plot(t,power_use)
    plt.title("power use for " + names[-1])
    plt.xlabel("Cycle")
    plt.ylabel("Power")
    plt.savefig("benchmarks/power_plots/power_use_" + names[-1] + ".pdf")
    plt.clf() 
    print("done!")

if __name__ == '__main__':
    main()