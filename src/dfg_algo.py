from cfg.staticfg.builder import CFGBuilder
import graphviz as gv
import ast
from cfg.staticfg.builder import CFGBuilder
from ast_utils import ASTUtils
import hardwareModel
benchmark = None
path = None
# format: node -> [symbol, id, write (true) or read (false)]
node_to_symbols = {}
# format: node -> {id -> dfg_node}
graphs = {}
cur_id = 0

class symbol:
    def __init__(self, value: str, num_id: str, write: bool):
        self.value = value
        self.num_id = num_id
        self.write = write

class Node:
    def __init__(self, value: str, operation: str):
        self.value = value
        self.operation = operation
        self.children = []
        self.parents = []
        self.order = 0

class Graph:
    def __init__(self, roots, id_to_Node):
        self.roots = roots
        self.id_to_Node = id_to_Node

# more generally, for one piece of data flowing to another location
# operand is the source of the data and operation is the destination
def process_operand(graph, operand, operand_num, operation_num, node):
    global node_to_symbols
    if type(operand) == ast.Name:
        make_node(graph, node, operand_num, operand.id, 'Read')
        make_edge(graph, node, operand_num, operation_num)
        node_to_symbols[node].append(symbol(operand.id, operand_num, False))
    elif type(operand) == ast.BinOp:
        op_id, value_ids = set_ids()
        eval_single_op(operand, graph, operation_num, value_ids, op_id, node)
    elif type(operand) == ast.Attribute:
        process_operand(graph, ast.Name(operand.attr), operand_num, operation_num, node)
    elif type(operand) == ast.List:
        for elt in operand.elts:
            process_operand(graph, elt, operand_num, operation_num, node)
            operand_num = set_id()
    elif type(operand) == ast.Subscript:
        # ignoring the slice for now
        process_operand(graph, operand.value, operand_num, operation_num, node)
    elif type(operand) == ast.Call:
        target = operand.func
        while type(target) == ast.Attribute or type(target) == ast.Subscript:
            # this should be a name
            target = target.value
        if type(target) == ast.Tuple: return
        target_id = set_id()
        make_node(graph, node, target_id, target.id, 'Write')
        for arg in operand.args:
            value_id = set_id()
            process_operand(graph, arg, value_id, target_id, node)
        node_to_symbols[node].append(symbol(target.id, target_id, True))
        graph.edge(target_id, operation_num)
    elif type(operand) == ast.Constant:
        graph.node(operand_num, str(operand.value))
        graph.edge(operand_num, operation_num)
    else: 
        print("unsupported operand type" + str(type(operand)))

# for now, only working with ast.binop
# add support for boolop
def eval_single_op(expr, graph, target_id, value_ids, op_id, node):
    global node_to_symbols
    sub_values = ASTUtils.get_sub_expr(expr)
    #print(sub_values)
    op_name = ASTUtils.operator_to_opname(sub_values[1])
    op_node = Node(hardwareModel.op2sym_map[op_name], op_name)
    graph.node(op_id, hardwareModel.op2sym_map[op_name])
    graphs[node].id_to_Node[op_id] = op_node
    graphs[node].roots.add(op_node)
    process_operand(graph, sub_values[0], value_ids[0], op_id, node)
    process_operand(graph, sub_values[2], value_ids[1], op_id, node)
    make_edge(graph, node, op_id, target_id)

def set_ids():
    global cur_id
    op_id = str(cur_id)
    value_ids = [str(cur_id+1), str(cur_id+2)]
    cur_id += 3
    return op_id, value_ids

def set_id():
    global cur_id
    val = str(cur_id)
    cur_id += 1
    return val

def eval_expr(expr, graph, node):
    if type(expr) == ast.Assign:
        target = expr.targets[0]
        while type(target) == ast.Attribute or type(target) == ast.Subscript:
            # this should be a name
            target = target.value
        if type(target) == ast.Tuple: return
        target_id = set_id()
        make_node(graph, node, target_id, target.id, 'Write')
        if type(expr.value) == ast.BinOp:
            op_id, value_ids = set_ids()
            eval_single_op(expr.value, graph, target_id, value_ids, op_id, node)
        else:
            operand_id = set_id()
            process_operand(graph, expr.value, operand_id, target_id, node)
        node_to_symbols[node].append(symbol(target.id, target_id, True))
    elif type(expr) == ast.AugAssign:
        target = expr.target
        while type(target) == ast.Attribute or type(target) == ast.Subscript:
            # this should be a name
            target = target.value
        if type(target) == ast.Tuple: return
        target_id = set_id()
        op_id, value_ids = set_ids()
        make_node(graph, node, target_id, target.id, 'Write')
        eval_single_op(ast.BinOp(expr.target, expr.op, expr.value), graph, target_id, value_ids, op_id, node)
        node_to_symbols[node].append(symbol(target.id, target_id, True))
    elif type(expr) == ast.Call:
        target = expr.func
        while type(target) == ast.Attribute or type(target) == ast.Subscript:
            # this should be a name
            target = target.value
        if type(target) == ast.Tuple: return
        target_id = set_id()
        make_node(graph, node, target_id, target.id, 'Write')
        for arg in expr.args:
            value_id = set_id()
            process_operand(graph, arg, value_id, target_id, node)
        node_to_symbols[node].append(symbol(target.id, target_id, True))
    elif type(expr) == ast.Expr:
        eval_expr(expr.value, graph, node)

# node for a non-literal
def make_node(graph, cfg_node, id, name, annotation):
    dfg_node = Node(name, annotation)
    graph.node(id, name + '\n' + annotation)
    graphs[cfg_node].roots.add(dfg_node)
    graphs[cfg_node].id_to_Node[id] = dfg_node

# edge for a non-literal
def make_edge(graph, node, source_id, target_id):
    source, target = graphs[node].id_to_Node[source_id], graphs[node].id_to_Node[target_id]
    graph.edge(source_id, target_id)
    target_node = graphs[node].id_to_Node[target_id]
    if target_node in graphs[node].roots: graphs[node].roots.remove(target_node)
    source.children.append(target)
    target.parents.append(source)

# first pass over the basic block
def dfg_per_node(node):
    graph = gv.Digraph()
    id = set_id()
    graph.node(id, "source code:\n" + node.get_source())
    for expr in node.statements:
        eval_expr(expr, graph, node)
    # walk backwards over statements, link reads to previous writes
    i = len(node_to_symbols[node])-1
    while i >= 0:
        if not node_to_symbols[node][i].write:
            j = i-1
            while j >= 0:
                if node_to_symbols[node][j].write and (node_to_symbols[node][j].value == node_to_symbols[node][i].value):
                    make_edge(graph, node, node_to_symbols[node][j].num_id, node_to_symbols[node][i].num_id)
                    break
                j -= 1
        i -= 1
    graph.render(path + 'pictures/' + benchmark + "_dfg_node_" + str(node.id), view = False, format='jpeg')
    return 0



def main_fn(path_in, benchmark_in):
    global benchmark, path, node_to_symbols, graphs
    benchmark, path = benchmark_in, path_in
    benchmark = benchmark[benchmark.rfind('/')+1:]
    cfg = CFGBuilder().build_from_file('main.c', path + 'models/' + benchmark)
    cfg.build_visual(path + 'pictures/' + benchmark, 'jpeg', show = False)
    for node in cfg:
        node_to_symbols[node] = []
        graphs[node] = Graph(set(), {})
        dfg_per_node(node)
        for root in graphs[node].roots:
            cur_node = root
            while True:
                #print(cur_node.value)
                if len(cur_node.children) == 0: break
                cur_node = cur_node.children[0]
            #print('')
    return cfg, graphs

if __name__ == "__main__":
    main_fn("")
