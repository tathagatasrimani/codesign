from cfg.staticfg.builder import CFGBuilder
import graphviz as gv
import ast
import astor
from cfg.staticfg.builder import CFGBuilder
from ast_utils import ASTUtils
import hardwareModel
benchmark = None
path = None
# format: node -> [symbol, id, write (true) or read (false)]
node_to_symbols = {}
# format: node -> {id -> dfg_node}
node_to_unroll = {}
unroll = False
graphs = {}
cur_id = 0

class symbol:
    def __init__(self, value: str, num_id: str, write: bool, read: bool):
        self.value = value
        self.num_id = num_id
        self.write = write
        self.read = read

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

def set_id():
    global cur_id
    val = str(cur_id)
    cur_id += 1
    return val

def eval_expr(expr, graph, node):
    global unroll
    if ASTUtils.isBoolOp(expr):
        print("visiting boolop")
        values = []
        for value in expr.values:
            values += eval_expr(value, graph, node)
        op_id = set_id()

        opname = ASTUtils.expr_to_opname(expr.op)
        make_node(graph, node, op_id, hardwareModel.op2sym_map[opname], None, opname) 
        for value in values:
            make_edge(graph, node, value, op_id)
        return [op_id]
    elif ASTUtils.isNamedExpr(expr):
        return
    elif ASTUtils.isBinOp(expr):
        print("visiting binop")
        left = eval_expr(expr.left, graph, node)
        right = eval_expr(expr.right, graph, node)
        op_id = set_id()
        opname = ASTUtils.expr_to_opname(expr.op)
        make_node(graph, node, op_id, hardwareModel.op2sym_map[opname], None, opname)
        if left:
            make_edge(graph, node, left[0], op_id)
        if right:
            make_edge(graph, node, right[0], op_id)
        return [op_id]
    elif ASTUtils.isUnaryOp(expr):
        print("visiting unaryop")
        value = eval_expr(expr.operand, graph, node)
        op_id = set_id()
        opname = ASTUtils.expr_to_opname(expr.op)
        make_node(graph, node, op_id, hardwareModel.op2sym_map[opname], None, opname)
        make_edge(graph, node, value[0], op_id)
        return [op_id]
    elif ASTUtils.isLambda(expr):
        return
    elif ASTUtils.isIfExp(expr):
        return
    elif ASTUtils.isDict(expr):
        return
    elif ASTUtils.isSet(expr):
        return
    elif ASTUtils.isListComp(expr):
        return
    elif ASTUtils.isSetComp(expr):
        return
    elif ASTUtils.isDictComp(expr):
        return
    elif ASTUtils.isGeneratorExp(expr):
        return
    elif ASTUtils.isAwait(expr):
        return
    elif ASTUtils.isYield(expr):
        return
    elif ASTUtils.isYieldFrom(expr):
        return
    elif ASTUtils.isCompare(expr):
        print("visiting compare")
        ids = []
        left = eval_expr(expr.left, graph, node)
        assert(len(expr.ops) == len(expr.comparators))
        for i in range(len(expr.ops)):
            comparator = eval_expr(expr.comparators[i], graph, node)
            op_id = set_id()
            ids.append(op_id)
            opname = ASTUtils.expr_to_opname(expr.ops[i])
            make_node(graph, node, op_id, hardwareModel.op2sym_map[opname], None, opname)
            make_edge(graph, node, left[0], op_id)
            make_edge(graph, node, comparator[0], op_id)
            left = comparator
        return ids
    elif ASTUtils.isCall(expr):
        print("visiting call")
        func_id = set_id()
        make_node(graph, node, func_id, astor.to_source(expr)[:-1], None, None)
        for arg in expr.args:
            arg_id = eval_expr(arg, graph, node)
            make_edge(graph, node, arg_id[0], func_id)
        return [func_id]
    elif ASTUtils.isFormattedValue(expr):
        return
    elif ASTUtils.isJoinedStr(expr):
        return
    elif ASTUtils.isConstant(expr):
        print("visiting constant")
        id = set_id()
        make_node(graph, node, id, str(expr.value), None, None)
        return [id]
    elif ASTUtils.isAttribute(expr):
        print("visiting attribute")
        if expr.attr == "start_unroll": unroll = True
        elif expr.attr == "stop_unroll": unroll = False
        if ASTUtils.isName(expr.value) or ASTUtils.isSubscript(expr.value):
            attr_id = set_id()
            make_node(graph, node, attr_id, astor.to_source(expr)[:-1], type(expr.ctx), "Regs")
            return [attr_id]
        else:
            target_id = eval_expr(expr.value, graph, node)
            attr_id = set_id()
            make_node(graph, node, attr_id, expr.attr, type(expr.ctx), "Regs")
            make_edge(graph, node, attr_id, target_id[0])
            return [attr_id]
    elif ASTUtils.isSubscript(expr):
        print("visiting subscript")
        # ignoring the index for now
        name_id = eval_expr(expr.value, graph, node)
        sub_id = set_id()
        make_node(graph, node, sub_id, astor.to_source(expr)[:-1], type(expr.ctx), "Regs")
        make_edge(graph, node, name_id[0], sub_id)
        return [sub_id]
    elif ASTUtils.isStarred(expr):
        return
    elif ASTUtils.isName(expr):
        print("visiting name")
        id = set_id()
        make_node(graph, node, id, expr.id, type(expr.ctx), "Regs")
        return [id]
    elif ASTUtils.isList(expr):
        print("visiting list")
        val = []
        for elem in expr.elts:
            val += eval_expr(elem, graph, node)
        if len(expr.elts) == 0:
            none_id = set_id()
            make_node(graph, node, none_id, "[]", None, None)
            val = [none_id]
        return val
    elif ASTUtils.isTuple(expr):
        print("visiting tuple")
        val = []
        for elem in expr.elts:
            val += eval_expr(elem, graph, node)
        return val
    elif ASTUtils.isSlice(expr):
        return

def eval_stmt(stmt, graph, node):
    global unroll
    if ASTUtils.isFunctionDef(stmt):
        for decorator in stmt.decorator_list:
            if ASTUtils.isName(decorator) and decorator.id == "unroll":
                unroll = True
                break
        else:
            unroll = False
        print(unroll)
    elif ASTUtils.isAsyncFunctionDef(stmt):
        return
    elif ASTUtils.isClassDef(stmt):
        return
    elif ASTUtils.isReturn(stmt):
        return
    elif ASTUtils.isDelete(stmt):
        return
    elif ASTUtils.isAssign(stmt):
        print("visiting assign")
        value_ids = eval_expr(stmt.value, graph, node)
        targets = eval_expr(stmt.targets[0], graph, node)
        if not targets or not value_ids: return
        if len(targets) > 1:
            if len(value_ids) == 1:
                for target in targets:
                    make_edge(graph, node, value_ids[0], target)
            elif len(targets) != len(value_ids):
                print("tuples of differing sizes")
                return
            else:
                for i in range(len(targets)):
                    make_edge(graph, node, value_ids[i], targets[i])
        else:
            for value_id in value_ids:
                make_edge(graph, node, value_id, targets[0])
    elif ASTUtils.isAugAssign(stmt):
        # note that target is a name
        print("visiting augassign")
        value_ids = eval_expr(stmt.value, graph, node)
        target = stmt.target
        while type(target) == ast.Attribute or type(target) == ast.Subscript:
            # this should be a name
            target = target.value
        if not target or not value_ids: return
        if len(value_ids) > 1:
            for value_id in value_ids:
                target_read_id = set_id()
                make_node(graph, node, target_read_id, target.id, ast.Load, "Regs")
                op_id = set_id()
                opname = ASTUtils.expr_to_opname(stmt.op)
                make_node(graph, node, op_id, hardwareModel.op2sym_map[opname], None, opname)
                make_edge(graph, node, value_id, op_id)
                make_edge(graph, node, target_read_id, op_id)
                target_write_id = eval_expr(stmt.target, graph, node)
                make_edge(graph, node, op_id, target_write_id[0])
        else:
            target_read_id = set_id()
            make_node(graph, node, target_read_id, target.id, ast.Load, "Regs")
            op_id = set_id()
            opname = ASTUtils.expr_to_opname(stmt.op)
            make_node(graph, node, op_id, hardwareModel.op2sym_map[opname], None, opname)
            make_edge(graph, node, value_ids[0], op_id)
            make_edge(graph, node, target_read_id, op_id)
            target_write_id = eval_expr(stmt.target, graph, node)
            make_edge(graph, node, op_id, target_write_id[0])
    elif ASTUtils.isAnnAssign(stmt):
        print("visiting annassign")
        target_id = eval_expr(stmt.target, graph, node)
        if not stmt.value:
            none_id = set_id()
            make_node(graph, node, none_id, "None", None, None)
            make_edge(graph, node, none_id, target_id[0])
        else:
            source_ids = eval_expr(stmt.value, graph, node)
            for source_id in source_ids:
                make_edge(graph, node, source_id, target_id[0])
        return target_id
    elif ASTUtils.isFor(stmt):
        print("visiting for")
        # target only evaluated once so going to ignore it here
        eval_expr(stmt.iter, graph, node)
    elif ASTUtils.isAsyncFor(stmt):
        print("visiting async for")
        eval_expr(stmt.iter, graph, node)
    elif ASTUtils.isWhile(stmt):
        print("visiting while")
        eval_expr(stmt.test, graph, node)
    elif ASTUtils.isIf(stmt):
        print("visiting if")
        eval_expr(stmt.test, graph, node)
    elif ASTUtils.isWith(stmt):
        return
    elif ASTUtils.isAsyncWith(stmt):
        return
    elif ASTUtils.isRaise(stmt):
        return
    elif ASTUtils.isTry(stmt):
        return
    elif ASTUtils.isAssert(stmt):
        return
    elif ASTUtils.isImport(stmt):
        return
    elif ASTUtils.isImportFrom(stmt):
        return
    elif ASTUtils.isGlobal(stmt):
        return
    elif ASTUtils.isNonlocal(stmt):
        return
    elif type(stmt) == ast.Expr:
        print("visiting expr")
        eval_expr(stmt.value, graph, node)
    elif ASTUtils.isCall(stmt):
        return

# node for a non-literal
def make_node(graph, cfg_node, id, name, ctx, opname):
    annotation = ""
    if ctx == ast.Load:
        annotation = "Read"
    elif ctx == ast.Store: # deal with Del if needed
        annotation = "Write"
    dfg_node = Node(name, opname)
    graph.node(id, name + '\n' + annotation)
    graphs[cfg_node].roots.add(dfg_node)
    graphs[cfg_node].id_to_Node[id] = dfg_node
    node_to_symbols[cfg_node].append(symbol(name, id, type(ctx) == ast.Store, type(ctx) == ast.Load))

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
    global node_to_unroll, unroll
    graph = gv.Digraph()
    graph.node(set_id(), "source code:\n" + node.get_source())
    for stmt in node.statements:
        eval_stmt(stmt, graph, node)
    node_to_unroll[node.id] = unroll
    # walk backwards over statements, link reads to previous writes
    i = len(node_to_symbols[node])-1
    while i >= 0:
        if node_to_symbols[node][i].read:
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
    global benchmark, path, node_to_symbols, graphs, node_to_unroll, unroll
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
    return cfg, graphs, node_to_unroll

if __name__ == "__main__":
    main_fn("")
