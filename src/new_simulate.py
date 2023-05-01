from schedule import schedule
from hls import HardwareModel
from cfg.ast_utils import ASTUtils
import ast
import hls
import math
import json

data = {}
cycles = 0
main_cfg = None
id_to_node = {}
path = '/Users/PatrickMcEwen/high_level_synthesis/venv/codesign/src/cfg/benchmarks/'
benchmark = 'simple'
data_path = []

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
    global cycles, data
    for i in range(max_cycles):
        print("This is during cycle " + str(cycles))
        print(hw_inuse)
        print("")
        # save current state of hardware to data array
        cur_data = ""
        for elem in hw_inuse:
            if len(hw_inuse[elem]) > 0:
                cur_data += elem + ": "
                count = 0
                for i in hw_inuse[elem]:
                    if i > 0:
                        count += 1
                cur_data += str(count) + "/" + str(len(hw_inuse[elem])) + " in use. || "
        data[cycles] = cur_data
        # simulate one cycle
        for elem in hw_inuse:
            for j in range(len(hw_inuse[elem])):
                if hw_inuse[elem][j] > 0:
                    hw_inuse[elem][j] = max(0, hw_inuse[elem][j] - 1)
        cycles += 1

def simulate(cfg, data_path, node_operations, hw_spec, first):
    global main_cfg, id_to_node
    cur_node = cfg.entryblock
    if first: 
        cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        main_cfg = cfg
    hw_inuse = {}
    for elem in hw_spec:
        hw_inuse[elem] = [0] * hw_spec[elem]
    print(hw_inuse)

    for elem in data_path:
        cur_node = id_to_node[elem[0]]
        for state in node_operations[cur_node]:
            hw_need = get_hw_need(state)
            print(hw_need)
            max_cycles = 0
            for elem in hw_need:
                cur_elem_count = hw_need[elem]
                if cur_elem_count == 0: continue
                if hw_spec[elem] < cur_elem_count: # this is just a basic condition, it might not work for every case
                    raise Exception("hardware specification insufficient to run program")
                cur_cycles_needed = math.ceil(cur_elem_count / hw_spec[elem]) * hls.latency[elem]
                print("cycles needed for " + elem + ": " + str(cur_cycles_needed) + ' (element count = ' + str(cur_elem_count) + ')')
                max_cycles = max(cur_cycles_needed, max_cycles)
                i = 0
                while cur_elem_count > 0:
                    hw_inuse[elem][i] += hls.latency[elem]
                    i = (i + 1) % hw_spec[elem]
                    cur_elem_count -= 1
            cycle_sim(hw_inuse, max_cycles)
        if len(cur_node.exits) != 0:
            # dumb version, just take the first exit node each time
            cur_node = cur_node.exits[0].target
        else:
            break
    return data

def main():
    cfg, graphs, node_operations = schedule()
    hw = HardwareModel(0, 0)
    hw.hw_allocated['Add'] = 2
    hw.hw_allocated['Regs'] = 5
    hw.hw_allocated['Mult'] = 1
    hw.hw_allocated['Sub'] = 1
    hw.hw_allocated['FloorDiv'] = 1
    hw.hw_allocated['Gt'] = 1
    for node in cfg:
        id_to_node[str(node.id)] = node
    # set up sequence of cfg nodes to visit
    with open('/Users/PatrickMcEwen/high_level_synthesis/venv/codesign/src/output.txt', 'r') as f:
        src = f.read()
        l = src.split('\n')
        for i in range(len(l)):
            l[i] = l[i].split()
        print(l)
        last_line = '-1'
        last_node = '-1'
        for item in l:
            if len(item) > 0 and (item[0] != last_node or item[1] == last_line):
                last_node = item[0]
                last_line = item[1]
                data_path.append(item)
        print(data_path)
        
    data = simulate(cfg, data_path, node_operations, hw.hw_allocated, first=True)
    print(data)
    text = json.dumps(data, indent=4)
    with open(path + 'json_data/' + benchmark, 'w') as fh:
        fh.write(text)

if __name__ == '__main__':
    main()