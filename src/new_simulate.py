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
state_graph_counter = 0

class HardwareSimulator():
    def __init__(self):
        self.memory_module = Memory(MEMORY_SIZE)
        self.data = {}
        self.cycles = 0
        self.main_cfg = None
        self.id_to_node = {}
        self.path = '/nfs/rsghome/pmcewen/codesign/src/' # change path variable for local computer
        self.data_path = []
        self.power_use = []
        self.node_intervals = []
        self.node_avg_power = {}
        self.unroll_at = {}
        self.vars_allocated = {}
        self.where_to_free = {}
        self.compute_element_to_node_id = {}
        self.compute_element_neighbors = {}
        self.memory_needed = 0
        self.cur_memory_size = 0
        self.new_graph = None
        self.mem_layers = 0
        self.transistor_size = 0
        self.pitch = 0
        self.cache_size = 0
        self.reads = 0
        self.writes = 0
        self.total_read_size = 0
        self.total_write_size = 0
        self.max_regs_inuse = 0
        self.max_mem_inuse = 0

    def find_nearest_mem_to_scale(self, num):
        if num < 512: return 512
        if num > 536870912: return 536870912
        return 2 ** math.ceil(math.log(num, 2))

    def make_node(self, graph, id, name, ctx, opname):
        annotation = ""
        if ctx == ast.Load or ctx == ast.Store:
            annotation = "Register"
        dfg_node = dfg_algo.Node(name, opname, id, memory_links=set())
        graph.gv_graph.node(id, name + '\n' + annotation)
        graph.roots.add(dfg_node)
        graph.id_to_Node[id] = dfg_node

    def make_edge(self, graph, source_id, target_id, annotation=""):
        source, target = graph.id_to_Node[source_id], graph.id_to_Node[target_id]
        graph.gv_graph.edge(source_id, target_id, label=annotation)
        target_node = graph.id_to_Node[target_id]
        if target_node in graph.roots: graph.roots.remove(target_node)
        source.children.append(target)
        target.parents.append(source)

    def get_hw_need(self, state, hw_spec):
        hw_need = HardwareModel(0,0,self.mem_layers, self.pitch, self.transistor_size, self.cache_size)
        mem_in_use = 0
        for op in state:
            if not op.operation: continue
            if op.operation != "Regs":
                #print(hw_spec[op.operation], op.operation, hw_need.hw_allocated[op.operation])
                compute_element_id = hw_need.hw_allocated[op.operation] % hw_spec.hw_allocated[op.operation]
                compute_id = self.compute_element_to_node_id[op.operation][compute_element_id]
                hw_op_node = self.new_graph.id_to_Node[compute_id]
                op.compute_id = compute_id
                #print(op.value)
                mem_in_use += self.process_compute_element(op, self.new_graph, hw_op_node, check_duplicate=True)
            hw_need.hw_allocated[op.operation] += 1
            hw_spec.compute_operation_totals[op.operation] += 1
        self.max_regs_inuse = min(hw_spec.hw_allocated["Regs"], max(self.max_regs_inuse, hw_need.hw_allocated["Regs"]))
        self.max_mem_inuse = max(self.max_mem_inuse, mem_in_use)
        return hw_need.hw_allocated

    def cycle_sim(self, hw_inuse, hw, max_cycles):
        node_power_sum = 0
        for i in range(max_cycles):
            self.power_use.append(0)
            #print("This is during cycle " + str(self.cycles))
            #print(hw_inuse)
            #print("")
            # save current state of hardware to data array
            cur_data = ""
            for elem in hw_inuse:
                power = hw.dynamic_power[elem]
                if elem == "Regs": power *= hw.power_scale[self.find_nearest_mem_to_scale(self.memory_needed)]
                if len(hw_inuse[elem]) > 0:
                    cur_data += elem + ": "
                    count = 0
                    for i in hw_inuse[elem]:
                        if i > 0:
                            count += 1
                            self.power_use[self.cycles] += power
                    self.power_use[self.cycles] += hw.leakage_power[elem] * len(hw_inuse[elem]) # passive power
                    cur_data += str(count) + "/" + str(len(hw_inuse[elem])) + " in use. || "
            self.data[self.cycles] = cur_data
            # simulate one cycle
            for elem in hw_inuse:
                for j in range(len(hw_inuse[elem])):
                    if hw_inuse[elem][j] > 0:
                        hw_inuse[elem][j] = max(0, hw_inuse[elem][j] - 1)
            node_power_sum += self.power_use[self.cycles]
            self.cycles += 1
        return node_power_sum

    def get_matching_bracket_count(self, name):
        if name.find('[') == -1:
            return 0
        bracket_count = 0
        name = name[name.find('[')+1:]
        bracket_depth = 1
        while len(name) > 0:
            front_ind = name.find('[')
            back_ind = name.find(']')
            if back_ind == -1: break
            if front_ind != -1 and front_ind < back_ind:
                bracket_depth += 1
                name = name[front_ind+1:]
            else:
                bracket_depth -= 1
                name = name[back_ind+1:]
                if bracket_depth == 0: bracket_count += 1
        return bracket_count

    def process_compute_element_neighbor(self, op, neighbor, graph, op_node, context, check_duplicate):
        name = neighbor.value
        mem_size = 0
        bracket_ind = name.find('[')
        bracket_count = self.get_matching_bracket_count(name)
        if bracket_ind != -1: name = name[:bracket_ind]
        #print(name, self.memory_module.locations)
        if name in self.memory_module.locations:
            mem_loc = self.memory_module.locations[name]
            mem_size = mem_loc.size
            #print(bracket_count, mem_loc.dims, neighbor.value)
            for i in range(bracket_count):
                mem_size /= mem_loc.dims[i]
            #print(mem_size)
            if check_duplicate:
                text = name + "\nlocation: " + str(mem_loc.location)
            else:
                text = name
            dfg_node_id = dfg_algo.set_id()
            #print(dfg_node_id)
            if check_duplicate:
                anno = "size: " + str(mem_size)
            else:
                anno = ""
            if (mem_loc.location, mem_loc.size) not in op_node.memory_links:
                self.make_node(graph, dfg_node_id, text, context, neighbor.operation)
                self.make_edge(graph, dfg_node_id, op_node.id, annotation=anno)
            op_node.memory_links.add((mem_loc.location, mem_loc.size))
            #print("node made for ", name)
        return mem_size

    def process_compute_element(self, op, graph, op_node, check_duplicate):
        parents = []
        for parent in op.parents:
            parents.append(parent.operation)
        #print(op.operation, parents)
        mem_in_use = 0
        for parent in op.parents:
            if not parent.operation: continue
            if parent.operation == "Regs":
                mem_in_use += self.process_compute_element_neighbor(op, parent, graph, op_node, ast.Load, check_duplicate)
            else:
                if not check_duplicate: continue
                #print(op.operation, parent.operation)
                parent_compute_id = parent.compute_id
                if parent_compute_id not in self.compute_element_neighbors[op_node.id]:
                    self.make_edge(graph, parent_compute_id, op_node.id, "")
                self.compute_element_neighbors[op_node.id].add(parent_compute_id)
                self.compute_element_neighbors[parent_compute_id].add(op_node.id)
        for child in op.children:
            if not child.operation: continue
            if child.operation == "Regs":
                mem_in_use += self.process_compute_element_neighbor(op, child, graph, op_node, ast.Store, check_duplicate)
        return mem_in_use
                
    def get_dims(self, arr):
        dims = []
        if arr[0][0] == '(': # processing tuple
            dims.append(int(arr[0][1:arr[0].find(',')]))
            if len(arr) > 2:
                for dim in arr[1:-1]:
                    dims.append(int(dim[:-1]))
            if len(arr) > 1:
                dims.append(int(arr[-1][:-1]))
        else: # processing array
            dims.append(int(arr[0][1:-1]))
            if len(arr) > 2:
                for dim in arr[1:-1]:
                    dims.append(int(dim[:-1]))
            if len(arr) > 1:
                dims.append(int(arr[-1][:-1]))
        return dims

    def process_memory_operation(self, mem_op):
        var_name = mem_op[2]
        size = int(mem_op[1])
        status = mem_op[0]
        if status == "malloc":
            dims = []
            if len(mem_op) > 3:
                dims = self.get_dims(mem_op[3:])
            self.memory_module.malloc(var_name, size, dims=dims)
        elif status == "free":
            self.memory_module.free(var_name)

    # adds all mallocs and frees to vectors, and finds the next cfg node in the data path,
    # returning the index of that node
    def find_next_data_path_index(self, i, mallocs, frees, data_path):
        pattern_seek = False
        max_iters = 1
        while len(self.data_path[i]) != 2:
            if len(self.data_path[i]) == 0: break
            elif len(self.data_path[i]) == 1:
                if self.data_path[i][0].startswith("pattern_seek"): 
                    pattern_seek = True
                    max_iters = int(self.data_path[i][0][self.data_path[i][0].rfind('_')+1:])
            if self.data_path[i][0] == "malloc": mallocs.append(self.data_path[i])
            elif self.data_path[i][0] == "free": frees.append(self.data_path[i])
            i += 1
            if i == len(self.data_path): break
        return i, pattern_seek, max_iters

    def simulate(self, cfg, node_operations, hw, graphs, first):
        global state_graph_counter
        cur_node = cfg.entryblock
        update_ind = 0
        if first: 
            cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
            self.main_cfg = cfg
        hw_inuse = {}
        for elem in hw.hw_allocated:
            hw_inuse[elem] = [0] * hw.hw_allocated[elem]
        #print(hw_inuse)
        i = 0
        frees = []
        mallocs = []
        #print(self.data_path)
        i, pattern_seek, max_iters = self.find_next_data_path_index(i, mallocs, frees, self.data_path)
        while i < len(self.data_path):
            next_ind, pattern_seek_next, max_iters_next = self.find_next_data_path_index(i+1, mallocs, frees, self.data_path)
            if i == len(self.data_path): break
            node_id = self.data_path[i][0]
            #print(node_id, self.memory_module.locations)
            #print(i)
            cur_node = self.id_to_node[node_id]
            self.node_intervals.append([node_id, [self.cycles, 0]])
            self.node_avg_power[node_id] = 0 # just reset because we will end up overwriting it
            start_cycles = self.cycles # for calculating average power
            iters = 0
            pattern_nodes = [node_id]
            pattern_mallocs, pattern_frees = [mallocs],  [frees]
            #print(pattern_seek)
            other_mallocs, other_frees = [], []
            if pattern_seek: 
                #print("entering pattern seek")
                pattern_seek = False
                j = next_ind
                while (not pattern_seek_next) and (j+1 < len(self.data_path)):
                    #print("hello")
                    next_node_id = self.data_path[j][0]
                    pattern_nodes.append(next_node_id)
                    pattern_seek = pattern_seek_next
                    j, pattern_seek_next, discard = self.find_next_data_path_index(j+1, other_mallocs, other_frees, self.data_path)
                    pattern_frees.append(other_frees)
                    pattern_mallocs.append(other_mallocs)
                    other_frees = []
                    other_mallocs = []
                if pattern_seek_next:
                    next_ind = j
                else:
                    pattern_nodes = [node_id]
                #print("found pattern")
                pattern_ind = 0
                while j+1 < len(self.data_path):
                    next_node_id = self.data_path[j][0]
                    if next_node_id != pattern_nodes[pattern_ind]: break
                    pattern_ind += 1
                    pattern_seek = pattern_seek_next
                    j, pattern_seek_next, discard = self.find_next_data_path_index(j+1, [], [], self.data_path)
                    if pattern_ind == len(pattern_nodes):
                        iters += 1
                        pattern_ind = 0
                        next_ind = j
                        if max_iters > 1 and iters+1 == max_iters: 
                            break
                if iters > 0: pattern_seek = True
            elif self.unroll_at[cur_node.id]:
                j = next_ind
                while j+1 < len(self.data_path):
                    next_node_id = self.data_path[j][0]
                    if next_node_id != node_id: break
                    iters += 1
                    pattern_seek = pattern_seek_next
                    j, pattern_seek_next, discard = self.find_next_data_path_index(j+1, [], [], self.data_path)
                next_ind = j
            #print(pattern_nodes, iters, next_ind)
            i = 0
            while i < len(pattern_nodes):
                #print("i: ", i)
                cur_node = self.id_to_node[pattern_nodes[i]]
                mallocs = pattern_mallocs[i]
                frees = pattern_frees[i]
                #print(mallocs, frees)
                for malloc in mallocs:
                    self.process_memory_operation(malloc)
                # try to unroll after pattern_seeking
                node_iters = iters
                if self.unroll_at[cur_node.id]:
                    #print(cur_node.id)
                    while i+1 != len(pattern_nodes) and pattern_nodes[i+1] == pattern_nodes[i]:
                        i += 1
                        node_iters += iters
                if self.unroll_at[cur_node.id] or pattern_seek:
                    for i in range(node_iters):
                        graph = dfg_algo.dfg_per_node(cur_node)
                        node_ops = schedule.schedule_one_node(graph, cur_node)
                        #print(node_operations[cur_node], node_ops)
                        for i in range(len(node_ops)):
                            node_operations[cur_node][i] = node_operations[cur_node][i] + node_ops[i]

                for state in node_operations[cur_node]:
                    # if unroll, take each operation in a state and create more of them
                    """if self.unroll_at[cur_node.id] or pattern_seek:
                        #print(iters, node_iters, len(state))
                        new_state = state.copy()
                        for op in state:
                            for j in range(node_iters):
                                new_state.append(op)
                        state = new_state"""
                    #print(state)
                    #print("new state")
                    hw_need = self.get_hw_need(state, hw)
                    state_graph_viz = gv.Graph()
                    state_graph = dfg_algo.Graph(set(), {}, state_graph_viz)
                    op_count = 0
                    for op in state:
                        if not op.operation or op.operation == "Regs": continue
                        op_count += 1
                        compute_id = dfg_algo.set_id()
                        self.make_node(state_graph, compute_id, hardwareModel.op2sym_map[op.operation], None, hardwareModel.op2sym_map[op.operation])
                        for parent in op.parents:
                            if parent.operation and parent.operation != "Regs":
                                parent_id = dfg_algo.set_id()
                                self.make_node(state_graph, parent_id, hardwareModel.op2sym_map[parent.operation], None, hardwareModel.op2sym_map[parent.operation])
                                self.make_edge(state_graph, parent_id, compute_id, "")
                        self.process_compute_element(op, state_graph, state_graph.id_to_Node[compute_id], check_duplicate=False)
                    #if op_count > 0:
                        #state_graph_viz.render(self.path + 'benchmarks/pictures/state_graphs/' + sys.argv[1][sys.argv[1].rfind('/')+1:] + '_' + str(state_graph_counter), view = True)
                    state_graph_counter += 1
                    #print(hw_need)
                    max_cycles = 0
                    for elem in hw_need:
                        latency = hw.latency[elem]
                        if elem == "Regs": latency *= hw.latency_scale[self.find_nearest_mem_to_scale(self.memory_needed)]
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
                    self.node_avg_power[node_id] += self.cycle_sim(hw_inuse, hw, max_cycles)
                i += 1
            if self.cycles - start_cycles > 0: self.node_avg_power[node_id] /= self.cycles - start_cycles
            self.node_intervals[-1][1][1] = self.cycles
            for free in frees:
                #print(free)
                if free[2] in self.memory_module.locations: 
                    self.process_memory_operation(free)
            mallocs = []
            frees = []
            i = next_ind
            pattern_seek = pattern_seek_next
            max_iters = max_iters_next
            update_ind += 1
            if (update_ind % 10 == 0): print(f"progress: {(i/len(self.data_path))*100} percent")
        #print("done with simulation")
        #self.new_graph.gv_graph.render(self.path + 'benchmarks/pictures/memory_graphs/' + sys.argv[1][sys.argv[1].rfind('/')+1:], view = True)
        return self.data

    def set_data_path(self):
        with open(self.path + '/instrumented_files/output.txt', 'r') as f:
            f_new = open(self.path + '/instrumented_files/output_free.txt', 'w+')
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
                    valid_names.add(item[2])
                if (item[-2] != "Read" and item[-2] != "Write"): continue
                var_name = item[0]
                if var_name not in valid_names: continue 
                if item[-2] == "Read": 
                    self.reads += 1
                    self.total_read_size += int(item[-1])
                    self.where_to_free[var_name] = i+1
                else: 
                    self.writes += 1
                    self.total_write_size += int(item[-1])
                    self.where_to_free[var_name] = i


            for i in range(len(split_lines)):
                item = split_lines[i]
                if not (len(item) >= 3 and item[0] == "malloc" and item[2] not in self.vars_allocated):
                    f_new.write(l[i] + '\n')
                vars_to_pop = []
                for var_name in self.where_to_free:
                    if self.where_to_free[var_name] == i:
                        var_size = self.vars_allocated[var_name]
                        f_new.write("free " + str(var_size) + " " + var_name + "\n")
                        self.data_path.append(["free", str(var_size), var_name])
                        vars_to_pop.append(var_name)
                        self.cur_memory_size -= var_size
                        self.vars_allocated.pop(var_name)
                for var_name in vars_to_pop:
                    self.where_to_free.pop(var_name)
                if len(item) == 2 and (item[0] != last_node or item[1] == last_line):
                    last_node = item[0]
                    last_line = item[1]
                    self.data_path.append(item)      
                elif len(item) == 1 and item[0].startswith('pattern_seek'):
                    self.data_path.append(item)
                elif len(item) >= 3 and item[0] == "malloc":
                    if item[2] in self.vars_allocated:
                        if int(item[1]) == self.vars_allocated[item[2]]: 
                            continue
                        else: 
                            f_new.write("free " + str(self.vars_allocated[item[2]]) + " " + var_name + "\n")
                            f_new.write(l[i] + '\n')
                        self.cur_memory_size -= int(self.vars_allocated[item[2]])
                    self.data_path.append(item)
                    self.vars_allocated[item[2]] = int(item[1])
                    #print(self.vars_allocated)
                    self.cur_memory_size += int(item[1])
                    self.memory_needed = max(self.memory_needed, self.cur_memory_size)
        #print(self.data_path)
        print("memory needed: ", self.memory_needed)

    def simulator_prep(self, benchmark):
        cfg, graphs, self.unroll_at = dfg_algo.main_fn(self.path, benchmark)
        node_operations = schedule.schedule(cfg, graphs, benchmark)
        self.set_data_path()
        for node in cfg: self.id_to_node[str(node.id)] = node
        #print(self.id_to_node)
        return cfg, graphs, node_operations

def main():
    benchmark = sys.argv[1]
    print(benchmark)
    simulator = HardwareSimulator()
    cfg, graphs, node_operations = simulator.simulator_prep(benchmark)
    simulator.transistor_size = 3 # in nm
    simulator.pitch = 100
    simulator.mem_layers = 2
    if simulator.memory_needed < 1000000:
        simulator.cache_size = 1
    elif simulator.memory_needed < 2000000:
        simulator.cache_size = 2
    elif simulator.memory_needed < 4000000:
        simulator.cache_size = 4
    elif simulator.memory_needed < 8000000:
        simulator.cache_size = 8
    else: 
        simulator.cache_size = 16
    hw = HardwareModel(0, 0, simulator.mem_layers, simulator.pitch, simulator.transistor_size, simulator.cache_size)
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
    simulator.new_graph = dfg_algo.Graph(set(), {}, new_gv_graph)
    for key in hw.hw_allocated:
        if key == "Regs": continue
        simulator.compute_element_to_node_id[key] = []
        for i in range(hw.hw_allocated[key]):
            compute_id = dfg_algo.set_id()
            simulator.make_node(simulator.new_graph, compute_id, hardwareModel.op2sym_map[key], None, hardwareModel.op2sym_map[key])
            simulator.compute_element_neighbors[compute_id] = set()
            simulator.compute_element_to_node_id[key].append(compute_id)
    
    data = simulator.simulate(cfg, node_operations, hw, graphs, True)
    print("total number of cycles: ", simulator.cycles)
    print("total energy (nJ): ", sum(simulator.power_use))
    print("total reads: ", simulator.reads)
    print("total read size: ", simulator.total_read_size)
    print("total writes: ", simulator.writes)
    print("total write size: ", simulator.total_write_size)
    print("total compute element usage: ", hw.compute_operation_totals)
    print("max regs in use: ", simulator.max_regs_inuse)
    print("max memory in use: ", simulator.max_mem_inuse)
    names = sys.argv[1].split('/')
    if len(sys.argv) < 3 or not sys.argv[2] == "notrace":
        text = json.dumps(data, indent=4)
        with open(simulator.path + 'benchmarks/json_data/' + names[-1], 'w') as fh:
            fh.write(text)
    t = []
    for i in range(len(simulator.power_use)):
        t.append(i)
    plt.plot(t,simulator.power_use)
    plt.title("power use for " + names[-1])
    plt.xlabel("Cycle")
    plt.ylabel("Power")
    plt.savefig("benchmarks/power_plots/power_use_" + names[-1] + ".pdf")
    plt.clf() 
    #print("done!")

if __name__ == '__main__':
    main()
