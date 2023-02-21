import json
import graphviz as gv
import re
from collections import deque
import ast

from cfg.staticfg.builder import CFGBuilder
from hls_instrument import instrument_and_run
from cfg.ast_utils import ASTUtils

path = '/Users/PatrickMcEwen/high_level_synthesis/venv/codesign/src/cfg/benchmarks/'
benchmark = 'simple'
expr_to_node = {}
func_ref = {}

op2sym_map = {
    "And": "and",
    "Or": "or",
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    "FloorDiv": "//",
    "Mod": "%",
    "LShift": "<<",
    "RShift": ">>",
    "BitOr": "|",
    "BitXor": "^",
    "BitAnd": "&",
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
    "IsNot": "!=",
    "USub": "-",
    "UAdd": "+",
    "Not": "!",
    "Invert": "~",
}
delimiters = (
    "+",
    "-",
    "*",
    "//",
    "%",
    "=",
    ">>",
    "<<",
    "<",
    "<=",
    ">",
    ">=",
    "!=",
    "~",
    "!",
    "^",
    "&",
)

latency = {
    "And": 1,
    "Or": 1,
    "Add": 4,
    "Sub": 4,
    "Mult": 5,
    "FloorDiv": 16,
    "Mod": 3,
    "LShift": 0.70,
    "RShift": 0.70,
    "BitOr": 0.06,
    "BitXor": 0.06,
    "BitAnd": 0.06,
    "Eq": 1,
    "NotEq": 1,
    "Lt": 1,
    "LtE": 1,
    "Gt": 1,
    "GtE": 1,
    "USub": 0.42,
    "UAdd": 0.42,
    "IsNot": 1,
    "Not": 0.06,
    "Invert": 0.06,
    "Regs": 1,
}
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

class HardwareModel:

    def __init__(self,id,bandwidth,loop_counts={},var_sizes={}):
        self.max_bw = bandwidth
        self.bw_avail = bandwidth

        self.loop_counts = loop_counts

        self.memory_cfgs = {}
        self.mem_state = {}
        for variable in self.memory_cfgs.keys():
            self.mem_state[variable]=False

        # number of non-memory elements allocated
        self.hw_allocated = {}
        self.hw_allocated["Regs"] = 0
        self.hw_allocated["Other"] = 0
        self.loop_variables = loop_counts
        self.var_sizes = var_sizes
        self.id = id


        for key in op2sym_map.keys():
                self.hw_allocated[key] = 0

        self.cycles = 0


    def print_stats(self):
        s = '''
        cycles={cycles}
        allocated={allocated}
        utilized={utilized}
        '''.format(cycles=self.cycles, \
                   allocated=str(self.hw_allocated))
        return s

    def eval_expr(self, expr):
        print(expr, type(expr))
        expr_to_node[expr] = self.id
        if type(expr) == ast.Name: 
            self.hw_allocated["Regs"] += 1
        else:
            name = ASTUtils.expr_to_opname(expr)
            if name:
                self.hw_allocated[name] += 1
                self.cycles += latency[name]
            else:
                for sub_expr in ASTUtils.get_sub_expr(expr):
                    self.eval_expr(sub_expr)
        if type(expr) == ast.FunctionDef:
            print(expr.body[0])
            func_ref[expr.name] = expr

def make_visual(cfg, models):
    graph = gv.Digraph()
    for node in cfg:
        hw = models[node.id].hw_allocated
        s = ""
        for i in hw:
            if hw[i] > 0:
                s += str(i) + ": " + str(hw[i]) + ", "
        if len(s) != 0:
            s = node.get_source() + "\n" + str(node.id) + ": [" + s[:-2] + "]" + "\n cycles: " + str(models[node.id].cycles)
        print(s)
        graph.node(str(node.id), s)
        for exit in node.exits:
            graph.edge(str(node.id), str(exit.target.id))
        print(node.func_calls, "these are the calls")
        for f in node.func_calls:
            if f in cfg.functioncfgs:
                graph.edge(str(node.id), str(cfg.functioncfgs[f].entryblock.id))
                print(node.id, cfg.functioncfgs[f].entryblock.id)
                for end in cfg.functioncfgs[f].finalblocks:
                    graph.edge(str(end.id), str(node.id))
    graph.render(path + 'pictures/' + benchmark + "_hw", view = True, format='jpeg')

def main():
    # note: must specify path to run the program, this is just an example path
    global path, benchmark, func_to_node, expr_to_node
    cfg = CFGBuilder().build_from_file('main.c', path + 'nonai_models/' + benchmark + '.py')
    cfg.build_visual(path + 'pictures/' + benchmark, 'jpeg', show = False)
    models = {}
    print(cfg.entryblock)
    print([block.id for block in cfg.__iter__()])
    print([cfg.functioncfgs[f] for f in cfg.functioncfgs], "hi")
    for node in cfg:
        #print([exit.target.id for exit in node.exits])
        print(node.func_calls, "func_calls")
        models[node.id] = HardwareModel(node.id, 0)
        for statement in node.statements:
            models[node.id].eval_expr(statement)
        print("Node", node.id, models[node.id].hw_allocated)
    make_visual(cfg, models)
    print(func_ref)
    print(expr_to_node)
    return 0

if __name__ == "__main__":
    main()