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
import graphviz as gv

MEMORY_SIZE = 1000000
memory_module = Memory(MEMORY_SIZE)

data = {}
cycles = 0
main_cfg = None
id_to_node = {}
path = '/nfs/pool0/pmcewen/codesign/codesign/src/' # change path variable for local computer
data_path = []
power_use = []
node_intervals = []
node_avg_power = {}
unroll_at = {}
vars_allocated = {}
where_to_free = {}
compute_element_to_node_id = {}
compute_element_neighbors = {}
memory_needed = 0
cur_memory_size = 0
new_graph = None
mem_layers = 0
transistor_size = 0
pitch = 0
cache_size = 0
reads = 0
writes = 0
total_read_size = 0
total_write_size = 0

def find_nearest_mem_to_scale(num):
    if num < 512: return 512
    if num > 536870912: return 536870912
    return 2 ** math.ceil(math.log(num, 2))

def make_node(graph, id, name, ctx, opname):
    global new_graph
    annotation = ""
    if ctx == ast.Load:
        annotation = "Read"
    elif ctx == ast.Store:
        annotation = "Write"
    dfg_node = dfg_algo.Node(name, opname, id, memory_links=set())
    graph.node(id, name + '\n' + annotation)
    new_graph.roots.add(dfg_node)
    new_graph.id_to_Node[id] = dfg_node

def make_edge(graph, source_id, target_id, annotation=""):
    global new_graph
    source, target = new_graph.id_to_Node[source_id], new_graph.id_to_Node[target_id]
    graph.edge(source_id, target_id, label=annotation)
    target_node = new_graph.id_to_Node[target_id]
    if target_node in new_graph.roots: new_graph.roots.remove(target_node)
    source.children.append(target)
    target.parents.append(source)

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

def get_hw_need(state, hw_spec):
    hw_need = HardwareModel(0,0,mem_layers, pitch, transistor_size, cache_size)
    for op in state:
        if not op.operation: continue
        if op.operation != "Regs":
            #print(hw_spec[op.operation], op.operation, hw_need.hw_allocated[op.operation])
            compute_element_id = hw_need.hw_allocated[op.operation] % hw_spec.hw_allocated[op.operation]
            compute_id = compute_element_to_node_id[op.operation][compute_element_id]
            hw_op_node = new_graph.id_to_Node[compute_id]
            op.compute_id = compute_id
            #print(op.value)
            process_compute_element(op, new_graph, hw_op_node)
        hw_need.hw_allocated[op.operation] += 1
        hw_spec.compute_operation_totals[op.operation] += 1
    return hw_need.hw_allocated

def cycle_sim(hw_inuse, hw, max_cycles):
    global cycles, data, power_use, memory_needed
    node_power_sum = 0
    for i in range(max_cycles):
        power_use.append(0)
        #print("This is during cycle " + str(cycles))
        #print(hw_inuse)
        #print("")
        # save current state of hardware to data array
        cur_data = ""
        for elem in hw_inuse:
            power = hw.dynamic_power[elem]
            if elem == "Regs": power *= hw.power_scale[find_nearest_mem_to_scale(memory_needed)]
            if len(hw_inuse[elem]) > 0:
                cur_data += elem + ": "
                count = 0
                for i in hw_inuse[elem]:
                    if i > 0:
                        count += 1
                        power_use[cycles] += power
                power_use[cycles] += hw.leakage_power[elem] * len(hw_inuse[elem]) # passive power
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

def process_compute_element_neighbor(op, neighbor, graph, op_node, context):
    global memory_module
    name = neighbor.value
    bracket_ind = name.find('[')
    if bracket_ind != -1: name = name[:bracket_ind]
    if name in memory_module.locations:
        mem_loc = memory_module.locations[name]
        text = name + "\nlocation: " + str(mem_loc.location)
        dfg_node_id = dfg_algo.set_id()
        #print(dfg_node_id)
        anno = "size: " + str(mem_loc.size)
        if (mem_loc.location, mem_loc.size) not in op_node.memory_links:
            make_node(graph.gv_graph, dfg_node_id, text, context, neighbor.operation)
            make_edge(graph.gv_graph, dfg_node_id, op_node.id, annotation=anno)
        op_node.memory_links.add((mem_loc.location, mem_loc.size))
        #print("node made for ", name)

def process_compute_element(op, graph, op_node):
    global memory_module
    parents = []
    for parent in op.parents:
        parents.append(parent.operation)
    #print(op.operation, parents)
    for parent in op.parents:
        if not parent.operation: continue
        if parent.operation == "Regs":
            process_compute_element_neighbor(op, parent, graph, op_node, ast.Load)
        else:
            #print(op.operation, parent.operation)
            parent_compute_id = parent.compute_id
            if parent_compute_id not in compute_element_neighbors[op_node.id]:
                make_edge(graph.gv_graph, parent_compute_id, op_node.id, "")
            compute_element_neighbors[op_node.id].add(parent_compute_id)
            compute_element_neighbors[parent_compute_id].add(op_node.id)
    for child in op.children:
        if not child.operation: continue
        if child.operation == "Regs":
            process_compute_element_neighbor(op, child, graph, op_node, ast.Store)
            

def process_memory_operation(var_name, size, status):
    global memory_module
    if status == "malloc":
        memory_module.malloc(var_name[:var_name.rfind('_')], size)
    elif status == "free":
        memory_module.free(var_name[:var_name.rfind('_')])

# adds all mallocs and frees to vectors, and finds the next cfg node in the data path,
# returning the index of that node
def find_next_data_path_index(i, mallocs, frees, data_path):
    pattern_seek = False
    max_iters = 1
    while len(data_path[i]) != 2:
        if len(data_path[i]) == 0: break
        elif len(data_path[i]) == 1:
            if data_path[i][0].startswith("pattern_seek"): 
                pattern_seek = True
                max_iters = int(data_path[i][0][data_path[i][0].rfind('_')+1:])
        if data_path[i][0] == "malloc": mallocs.append(data_path[i])
        elif data_path[i][0] == "free": frees.append(data_path[i])
        i += 1
        if i == len(data_path): break
    return i, pattern_seek, max_iters

def simulate(cfg, node_operations, hw, graphs, first):
    global main_cfg, id_to_node, unroll_at, memory_module, data_path, new_graph, memory_needed
    cur_node = cfg.entryblock
    if first: 
        cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        main_cfg = cfg
    hw_inuse = {}
    for elem in hw.hw_allocated:
        hw_inuse[elem] = [0] * hw.hw_allocated[elem]
    #print(hw_inuse)
    i = 0
    frees = []
    mallocs = []
    i, pattern_seek, max_iters = find_next_data_path_index(i, mallocs, frees, data_path)
    while i < len(data_path):
        next_ind, pattern_seek_next, max_iters_next = find_next_data_path_index(i+1, mallocs, frees, data_path)
        if i == len(data_path): break
        node_id = data_path[i][0]
        #print(node_id, memory_module.locations)
        #print(i)
        cur_node = id_to_node[node_id]
        node_intervals.append([node_id, [cycles, 0]])
        node_avg_power[node_id] = 0 # just reset because we will end up overwriting it
        start_cycles = cycles # for calculating average power
        iters = 0
        pattern_nodes = [node_id]
        pattern_mallocs, pattern_frees = [mallocs],  [frees]
        #print(pattern_seek)
        other_mallocs, other_frees = [], []
        if pattern_seek: 
            #print("entering pattern seek")
            pattern_seek = False
            j = next_ind
            while not pattern_seek_next:
                #print("hello")
                if len(data_path) <= j: break
                next_node_id = data_path[j][0]
                pattern_nodes.append(next_node_id)
                pattern_seek = pattern_seek_next
                j, pattern_seek_next, discard = find_next_data_path_index(j+1, other_mallocs, other_frees, data_path)
                pattern_frees.append(other_frees)
                pattern_mallocs.append(other_mallocs)
                other_frees.clear()
                other_mallocs.clear()
            if pattern_seek_next:
                next_ind = j
            else:
                pattern_nodes = [node_id]
            #print("found pattern")
            pattern_ind = 0
            while j < len(data_path):
                next_node_id = data_path[j][0]
                if next_node_id != pattern_nodes[pattern_ind]: break
                pattern_ind += 1
                pattern_seek = pattern_seek_next
                j, pattern_seek_next, discard = find_next_data_path_index(j+1, [], [], data_path)
                if pattern_ind == len(pattern_nodes):
                    iters += 1
                    pattern_ind = 0
                    next_ind = j
                    if max_iters > 1 and iters+1 == max_iters: 
                        break
            if iters > 0: pattern_seek = True
        elif unroll_at[cur_node.id]:
            j = next_ind
            while True:
                if len(data_path) <= j: break
                next_node_id = data_path[j][0]
                if next_node_id != node_id: break
                iters += 1
                pattern_seek = pattern_seek_next
                j, pattern_seek_next, discard = find_next_data_path_index(j+1, [], [], data_path)
            next_ind = j
        #print(pattern_nodes, iters, next_ind)
        i = 0
        while i < len(pattern_nodes):
            #print("i: ", i)
            cur_node = id_to_node[pattern_nodes[i]]
            mallocs = pattern_mallocs[i]
            frees = pattern_frees[i]
            #print(mallocs, frees)
            for malloc in mallocs:
                process_memory_operation(malloc[2], int(malloc[1]), malloc[0])
            # try to unroll after pattern_seeking
            node_iters = iters
            if unroll_at[cur_node.id]:
                while i+1 != len(pattern_nodes) and pattern_nodes[i+1] == pattern_nodes[i]:
                    i += 1
                    node_iters += iters
            for state in node_operations[cur_node]:
                # if unroll, take each operation in a state and create more of them
                if unroll_at[cur_node.id] or pattern_seek:
                    #print(iters, node_iters, len(state))
                    new_state = state.copy()
                    for op in state:
                        for j in range(node_iters):
                            new_state.append(op)
                    state = new_state
                #print(state)
                #print("new state")
                for node in state: 
                    parents = []
                    for parent in node.parents:
                        parents.append(parent.operation)
                    #print(node.operation, parents)
                hw_need = get_hw_need(state, hw)
                #print(hw_need)
                max_cycles = 0
                for elem in hw_need:
                    latency = hw.latency[elem]
                    if elem == "Regs": latency *= hw.latency_scale[find_nearest_mem_to_scale(memory_needed)]
                    cur_elem_count = hw_need[elem]
                    if cur_elem_count == 0: continue
                    if hw.hw_allocated[elem] == 0 and cur_elem_count > 0:
                        raise Exception("hardware specification insufficient to run program")
                    cur_cycles_needed = int(math.ceil(cur_elem_count / hw.hw_allocated[elem]) * latency)
                    #print("cycles needed for " + elem + ": " + str(cur_cycles_needed) + ' (element count = ' + str(cur_elem_count) + ')')
                    max_cycles = max(cur_cycles_needed, max_cycles)
                    j = 0
                    while cur_elem_count > 0:
                        hw_inuse[elem][j] += latency
                        j = (j + 1) % hw.hw_allocated[elem]
                        cur_elem_count -= 1
                node_avg_power[node_id] += cycle_sim(hw_inuse, hw, max_cycles)
            i += 1
        if cycles - start_cycles > 0: node_avg_power[node_id] /= cycles - start_cycles
        node_intervals[-1][1][1] = cycles
        for free in frees:
            #print(free)
            if free[2][:free[2].rfind('_')] in memory_module.locations: 
                process_memory_operation(free[2], int(free[1]), free[0])
        mallocs.clear()
        frees.clear()
        i = next_ind
        pattern_seek = pattern_seek_next
        max_iters = max_iters_next
    print("done with simulation")
    #new_graph.gv_graph.render(path + 'benchmarks/pictures/memory_graphs/' + sys.argv[1][sys.argv[1].rfind('/')+1:], view = True)
    return data

def set_data_path():
    global path, data_path, cur_memory_size, vars_allocated, where_to_free, memory_needed, reads, writes, total_read_size, total_write_size
    with open(path + '/instrumented_files/output.txt', 'r') as f:
        f_new = open(path + '/instrumented_files/output_free.txt', 'w+')
        src = f.read()
        l = src.split('\n')
        split_lines = []
        for i in range(len(l)):
            split_lines.append(l[i].split())
        #print(l)
        last_line = '-1'
        last_node = '-1'
        valid_names = set()
        for i in range(len(split_lines)):
            item = split_lines[i]
            if len(item) < 2: continue
            if item[0] == "malloc":
                valid_names.add(item[-1])
            if (item[-2] != "Read" and item[-2] != "Write"): continue
            var_name = item[0]
            if var_name not in valid_names: continue 
            if item[-2] == "Read": 
                reads += 1
                total_read_size += int(item[-1])
                where_to_free[var_name] = i+1
            else: 
                writes += 1
                total_write_size += int(item[-1])
                where_to_free[var_name] = i


        for i in range(len(split_lines)):
            item = split_lines[i]
            if not (len(item) == 3 and item[0] == "malloc" and item[2] not in vars_allocated):
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
            elif len(item) == 1 and item[0].startswith('pattern_seek'):
                data_path.append(item)
            elif len(item) == 3 and item[0] == "malloc":
                if item[2] in vars_allocated:
                    if int(item[1]) == vars_allocated[item[2]]: 
                        continue
                    else: 
                        f_new.write("free " + str(vars_allocated[item[2]]) + " " + var_name + "\n")
                        f_new.write(l[i] + '\n')
                    cur_memory_size -= int(vars_allocated[item[2]])
                data_path.append(item)
                vars_allocated[item[2]] = int(item[1])
                #print(vars_allocated)
                cur_memory_size += int(item[1])
                memory_needed = max(memory_needed, cur_memory_size)
    #print(data_path)
    print("memory needed: ", memory_needed)

def simulator_prep(benchmark):
    global unroll_at, id_to_node
    cfg, graphs, unroll_at = dfg_algo.main_fn(path + 'benchmarks/', benchmark)
    node_operations = schedule.schedule(cfg, graphs, benchmark)
    set_data_path()
    for node in cfg: id_to_node[str(node.id)] = node
    return cfg, graphs, node_operations

def main():
    global power_use, new_graph, transistor_size, pitch, mem_layers, memory_needed, cache_size, reads, writes, total_read_size, total_write_size
    benchmark = sys.argv[1]
    print(benchmark)
    cfg, graphs, node_operations = simulator_prep(benchmark)
    transistor_size = 3 # in nm
    pitch = 100
    mem_layers = 2
    if memory_needed < 1000000:
        cache_size = 1
    elif memory_needed < 2000000:
        cache_size = 2
    elif memory_needed < 4000000:
        cache_size = 4
    elif memory_needed < 8000000:
        cache_size = 8
    else: 
        cache_size = 16
    hw = HardwareModel(0, 0, mem_layers, pitch, transistor_size, cache_size)
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

    area = 0
    for key in hw.hw_allocated:
        area += hw.hw_allocated[key] * hw.area[key]
    print("compute area: ", area)

    new_gv_graph = gv.Graph()
    new_graph = dfg_algo.Graph(set(), {}, new_gv_graph)
    for key in hw.hw_allocated:
        if key == "Regs": continue
        compute_element_to_node_id[key] = []
        for i in range(hw.hw_allocated[key]):
            compute_id = dfg_algo.set_id()
            make_node(new_graph.gv_graph, compute_id, hardwareModel.op2sym_map[key], None, hardwareModel.op2sym_map[key])
            compute_element_neighbors[compute_id] = set()
            compute_element_to_node_id[key].append(compute_id)
    
    data = simulate(cfg, node_operations, hw, graphs, True)
    print("total number of cycles: ", cycles)
    print("total energy (nJ): ", sum(power_use))
    print("total reads: ", reads)
    print("total read size: ", total_read_size)
    print("total writes: ", writes)
    print("total write size: ", total_write_size)
    print("total compute element usage: ", hw.compute_operation_totals)
    names = sys.argv[1].split('/')
    if len(sys.argv) < 3 or not sys.argv[2] == "notrace":
        text = json.dumps(data, indent=4)
        with open(path + 'benchmarks/json_data/' + names[-1], 'w') as fh:
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