import graphviz as gv
import ast
import astor

from staticfg.builder import CFGBuilder
from ast_utils import ASTUtils
from config_dicts import op2sym_map

benchmark = None
path = None
# format: node -> [symbol, id, write (true) or read (false)]
node_to_symbols = {}
# format: node -> {id -> dfg_node}
node_to_unroll = {}
unroll = False
# graphs is a dictionary of cfg_node to dfg_algo.Graph objects
graphs = {}
cur_id = 0


class symbol:
    def __init__(self, value: str, num_id: str, write: bool, read: bool):
        self.value = value
        self.num_id = num_id
        self.write = write
        self.read = read


class Node:
    def __init__(
        self, value: str, operation: str, id, memory_links=None, compute_id=None
    ):
        self.value = value
        self.operation = operation
        self.memory_links = memory_links
        self.children = []
        self.parents = []
        self.order = 0
        self.id = id
        self.compute_id = compute_id

    def __str__(self):
        return f"dfg Node {self.id}: {self.value}, op: {self.operation}, memory_links: {self.memory_links}, compute_id: {self.compute_id}"


class Graph:
    def __init__(self, roots, id_to_Node, gv_graph):
        """
        roots - set
        id_to_Node - dict
        gv_graph - graphviz.Digraph
        """
        self.roots = roots
        self.id_to_Node = id_to_Node
        self.gv_graph = gv_graph
        self.max_id = 0

    def set_gv_graph(self, graph):
        self.gv_graph = graph
    
    def __str__(self):
        return f"dfg Graph: roots -> {[str(node) for node in self.roots]}\nid_to_node->{self.id_to_Node}\nmax_id->{self.max_id}"


def set_id():
    global cur_id
    val = str(cur_id)
    cur_id += 1
    return val


def eval_expr(expr, graph, node):
    global unroll
    if ASTUtils.isBoolOp(expr):
        # print("visiting boolop")
        values = []
        for value in expr.values:
            values += eval_expr(value, graph, node)
        op_id = set_id()

        opname = ASTUtils.expr_to_opname(expr.op)
        make_node(graph, node, op_id, op2sym_map[opname], None, opname)
        for value in values:
            make_edge(graph, node, value, op_id)
        return [op_id]
    elif ASTUtils.isNamedExpr(expr):
        return
    elif ASTUtils.isBinOp(expr):
        # print("visiting binop")
        left = eval_expr(expr.left, graph, node)
        right = eval_expr(expr.right, graph, node)
        op_id = set_id()
        opname = ASTUtils.expr_to_opname(expr.op)
        make_node(graph, node, op_id, op2sym_map[opname], None, opname)
        if left:
            make_edge(graph, node, left[0], op_id)
        if right:
            make_edge(graph, node, right[0], op_id)
        return [op_id]
    elif ASTUtils.isUnaryOp(expr):
        # print("visiting unaryop")
        value = eval_expr(expr.operand, graph, node)
        op_id = set_id()
        opname = ASTUtils.expr_to_opname(expr.op)
        make_node(graph, node, op_id, op2sym_map[opname], None, opname)
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
        # print("visiting compare")
        ids = []
        left = eval_expr(expr.left, graph, node)
        assert len(expr.ops) == len(expr.comparators)
        for i in range(len(expr.ops)):
            comparator = eval_expr(expr.comparators[i], graph, node)
            op_id = set_id()
            ids.append(op_id)
            opname = ASTUtils.expr_to_opname(expr.ops[i])
            make_node(
                graph, node, op_id, op2sym_map[opname], None, opname
            )
            make_edge(graph, node, left[0], op_id)
            make_edge(graph, node, comparator[0], op_id)
            left = comparator
        return ids
    elif ASTUtils.isCall(expr):
        # print("visiting call")
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
        # print("visiting constant")
        id = set_id()
        make_node(graph, node, id, str(expr.value), None, None)
        return [id]
    elif ASTUtils.isAttribute(expr):
        # print("visiting attribute")

        if expr.attr == "start_unroll":
            unroll = True
        elif expr.attr == "stop_unroll":
            unroll = False
        if ASTUtils.isName(expr.value) or ASTUtils.isSubscript(expr.value):
            attr_id = set_id()
            make_node(
                graph, node, attr_id, astor.to_source(expr)[:-1], type(expr.ctx), "Regs"
            )
            return [attr_id]
        else:
            target_id = eval_expr(expr.value, graph, node)
            attr_id = set_id()
            make_node(graph, node, attr_id, expr.attr, type(expr.ctx), "Regs")
            make_edge(graph, node, attr_id, target_id[0])
            return [attr_id]
    elif ASTUtils.isSubscript(expr):
        # print("visiting subscript")
        # ignoring the index for now
        # name_id = eval_expr(expr.value, graph, node)
        sub_id = set_id()
        make_node(
            graph, node, sub_id, astor.to_source(expr)[:-1], type(expr.ctx), "Regs"
        )
        # make_edge(graph, node, name_id[0], sub_id)
        return [sub_id]
    elif ASTUtils.isStarred(expr):
        return
    elif ASTUtils.isName(expr):
        # print("visiting name")
        id = set_id()
        make_node(graph, node, id, expr.id, type(expr.ctx), "Regs")
        return [id]
    elif ASTUtils.isList(expr):
        # print("visiting list")
        val = []
        for elem in expr.elts:
            val += eval_expr(elem, graph, node)
        if len(expr.elts) == 0:
            none_id = set_id()
            make_node(graph, node, none_id, "[]", None, None)
            val = [none_id]
        return val
    elif ASTUtils.isTuple(expr):
        # print("visiting tuple")
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
        # print(unroll)
    elif ASTUtils.isAsyncFunctionDef(stmt):
        return
    elif ASTUtils.isClassDef(stmt):
        return
    elif ASTUtils.isReturn(stmt):
        return
    elif ASTUtils.isDelete(stmt):
        return
    elif ASTUtils.isAssign(stmt):
        # print("visiting assign")
        value_ids = eval_expr(stmt.value, graph, node)
        targets = eval_expr(stmt.targets[0], graph, node)
        if not targets or not value_ids:
            return
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
        # print("visiting augassign")
        value_ids = eval_expr(stmt.value, graph, node)
        target = stmt.target
        if not target or not value_ids:
            return
        stmt.target.ctx = ast.Load()
        target_read_id = eval_expr(stmt.target, graph, node)
        stmt.target.ctx = ast.Store()
        if len(value_ids) > 1:
            for value_id in value_ids:
                op_id = set_id()
                opname = ASTUtils.expr_to_opname(stmt.op)
                make_node(
                    graph, node, op_id, op2sym_map[opname], None, opname
                )
                make_edge(graph, node, value_id, op_id)
                make_edge(graph, node, target_read_id[0], op_id)
                target_write_id = eval_expr(stmt.target, graph, node)
                make_edge(graph, node, op_id, target_write_id[0])
        else:
            op_id = set_id()
            opname = ASTUtils.expr_to_opname(stmt.op)
            make_node(
                graph, node, op_id, op2sym_map[opname], None, opname
            )
            make_edge(graph, node, value_ids[0], op_id)
            make_edge(graph, node, target_read_id[0], op_id)
            target_write_id = eval_expr(stmt.target, graph, node)
            make_edge(graph, node, op_id, target_write_id[0])
    elif ASTUtils.isAnnAssign(stmt):
        # print("visiting annassign")
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
        # print("visiting for")
        # target only evaluated once so going to ignore it here
        eval_expr(stmt.iter, graph, node)
    elif ASTUtils.isAsyncFor(stmt):
        # print("visiting async for")
        eval_expr(stmt.iter, graph, node)
    elif ASTUtils.isWhile(stmt):
        # print("visiting while")
        eval_expr(stmt.test, graph, node)
    elif ASTUtils.isIf(stmt):
        # print("visiting if")
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
        # print("visiting expr")
        eval_expr(stmt.value, graph, node)
    elif ASTUtils.isCall(stmt):
        return


# node for a non-literal
def make_node(graph, cfg_node, id, name, ctx, opname):
    annotation = ""
    if ctx == ast.Load:
        annotation = "Read"
    elif ctx == ast.Store:  # deal with Del if needed
        annotation = "Write"
    dfg_node = Node(name, opname, id)
    graph.node(id, name + "\n" + annotation)
    graphs[cfg_node].roots.add(dfg_node)
    graphs[cfg_node].id_to_Node[id] = dfg_node
    node_to_symbols[cfg_node].append(
        symbol(name, id, ctx == ast.Store, ctx == ast.Load)
    )


# edge for a non-literal
def make_edge(graph, node, source_id, target_id, annotation=""):
    source, target = (
        graphs[node].id_to_Node[source_id],
        graphs[node].id_to_Node[target_id],
    )
    graph.edge(source_id, target_id, label=annotation)
    target_node = graphs[node].id_to_Node[target_id]
    if target_node in graphs[node].roots:
        graphs[node].roots.remove(target_node)
    source.children.append(target)
    target.parents.append(source)


# first pass over the basic block
def dfg_per_node(node):
    global node_to_unroll, unroll, graphs
    graph = gv.Digraph()
    graphs[node] = Graph(set(), {}, None)
    node_to_symbols[node] = []
    graphs[node].set_gv_graph(graph)
    graph.node(set_id(), "source code:\n" + node.get_source())
    for stmt in node.statements:
        eval_stmt(stmt, graph, node)
    node_to_unroll[node.id] = unroll
    # walk backwards over statements, link reads to previous writes
    i = len(node_to_symbols[node]) - 1
    while i >= 0:
        # print(node_to_symbols[node][i].read, node_to_symbols[node][i].write, node_to_symbols[node][i].value)
        if node_to_symbols[node][i].read:
            name = node_to_symbols[node][i].value
            if name.find("[") != -1:
                name = name[: name.find("[")]
            j = i - 1
            while j >= 0:
                other_name = node_to_symbols[node][j].value
                if other_name.find("[") != -1:
                    other_name = other_name[: other_name.find("[")]
                # print(name, other_name)
                if node_to_symbols[node][j].write and (other_name == name):
                    make_edge(
                        graph,
                        node,
                        node_to_symbols[node][j].num_id,
                        node_to_symbols[node][i].num_id,
                    )
                    break
                j -= 1
        i -= 1
    # graph.render(path + '/benchmarks/pictures/' + benchmark + "_dfg_node_" + str(node.id), view = False)
    return graphs[node]


def main_fn(path_in, benchmark_in):
    global benchmark, path, node_to_symbols, graphs, node_to_unroll, unroll
    benchmark, path = benchmark_in, path_in
    benchmark = benchmark[benchmark.rfind("/") + 1 :]
    cfg = CFGBuilder().build_from_file(
        "main.c", path + "/instrumented_files/xformedname-" + benchmark
    )
    cfg.build_visual(path + "/benchmarks/pictures/" + benchmark, "jpeg", show=False)
    for node in cfg:
        dfg_per_node(node)
        for root in graphs[node].roots:
            cur_node = root
            while True:
                # print(cur_node.value)
                if len(cur_node.children) == 0:
                    break
                cur_node = cur_node.children[0]
            # print('')
    return cfg, graphs, node_to_unroll
