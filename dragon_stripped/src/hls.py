import ast
import astor
import numpy as np
import re
import src.cfg_model as cfg_model
from src.ast_utils import ASTUtils
import sys


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

#hw_allocated = {}
#memory_cfgs = {}
#hw_utilized = {}
#bw_avail = 0
#mem_state = {}
#for variable in memory_cfgs.keys():
#    mem_state[variable]=False
#    print(variable)
#cycles = 0
#hw_allocated["Regs"] = 0
#hw_utilized["Regs"] = 0
class HLSOp:

    def __init__(self):
        pass


class HardwareModel:

    def __init__(self,bandwidth,loop_counts={},var_sizes={}):
        self.max_bw = bandwidth
        self.bw_avail = bandwidth

        self.loop_counts = loop_counts

        self.memory_cfgs = {}
        self.mem_state = {}
        for variable in self.memory_cfgs.keys():
            self.mem_state[variable]=False

        # number of non-memory elements allocated
        self.hw_allocated = {}
        self.hw_utilized = {}
        self.hw_allocated["Regs"] = 0
        self.hw_utilized["Regs"] = 0
        self.hw_allocated["Other"] = 0
        self.hw_utilized["Other"] = 0
        self.loop_variables = loop_counts
        self.var_sizes = var_sizes
        self.blocks = {}


        for key in op2sym_map.keys():
                self.hw_allocated[key] = 0
                self.hw_utilized[key] = 0

        self.cycles = 0


    def print_stats(self):
        s = '''
        cycles={cycles}
        allocated={allocated}
        utilized={utilized}
        '''.format(cycles=self.cycles, \
                   allocated=str(self.hw_allocated), \
                   utilized=str(self.hw_utilized))
        return s


    def get_unroll_count(self,varname):
        assert(varname in self.loop_variables)
        iters = self.loop_variables[varname]
        if len(set(iters)) > 1:
            return 1
        else:
            return list(set(iters))[0]


    def get_loop_count(self,varname):
        assert(varname in self.loop_variables)
        return max(self.loop_variables[varname])


    def has_global_var(self,var):
        return len(self._get_global_var_size_helper(var))  > 0

    def get_global_var_size(self,var):
        vals = self._get_global_var_size_helper(var)
        if len(vals ) >= 1:
            return max(vals)
        else:
            0.0

    def _get_global_var_size_helper(self,var):
        vals = []
        for lv,lst in self.var_sizes.items():
            for (v,siz) in lst:
                if v == var:
                    vals.append(siz)

        return vals

    def add_execution_block(self,node,blk):
        assert(isinstance(blk, ExecutionBlock))
        self.blocks[ASTUtils.get_identifier(node)] = blk

        self.cycles += blk.compute_cycles+blk.mem_cycles

        for compute_el,count in blk.hardware_elements():
            self.hw_allocated[compute_el] = max(count,self.hw_allocated[compute_el])

        for compute_el,count in blk.hardware_elements():
            self.hw_utilized[compute_el] += count

    def get_latency(self,expr : ast.AST):
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
            "Regs": 1
        }
        name = ASTUtils.expr_to_opname(expr)
        if name is None:
            return None,0.0
        else:
            return name,latency[name]

    @staticmethod
    def parse_expr(text):
        pass

class ExecutionBlock:

    def __init__(self,max_bw):
        self.hw_need = {}
        #self.compute_cycles = 0
        #self.mem_cycles = 0
        self.hw_need["Regs"] = 0
        self.hw_need["Other"] = 0
        for key in op2sym_map.keys():
            self.hw_need[key] = 0

        self._freeze= False
        self.read_footprint = 0
        self.max_bw = max_bw
        self.scale_hardware= 1
        self.scale_cycle = 1
        self.compute_cycles = 0
        self.mem_cycles = 0

    def copy(self):
        e = ExecutionBlock(self.max_bw)
        #e.compute_cycles = self.compute_cycles
        #e.mem_cycles = self.mem_cycles
        e.read_footprint = self.read_footprint
        e.hw_need = dict(self.hw_need)
        e.scale_hardware= self.scale_hardware
        e.scale_cycle = self.scale_cycle
        return e

    def scale_cycles(self,amt:HLSOp):
        assert(not self.frozen)
        self.scale_cycle *= amt
        return self

    def hardware_elements(self):
        for k,v in self.hw_need.items():
            yield k,v

    def scale_hardware_elements(self,amt:float):
        assert(not self.frozen)
        self.scale_hardware *= amt

    @property
    def frozen(self):
        return self._freeze

    def add_op(self,model:HardwareModel, op: ast.AST):
        assert(not self.frozen)
        blk_name,lat = model.get_latency(op)
        self.compute_cycles += lat
        if blk_name is None:
            self.hw_need["Other"] += 1
        else:
            self.hw_need[blk_name] += 1

    def load_from_memory(self,model:HardwareModel, varname: str):
        assert(not self.frozen)
        if model.has_global_var(varname):
            siz = model.get_global_var_size(varname)
            # assumes one register per variable in memory
            self.read_footprint += siz
        else:
            print("[warn] could not find variable <%s>" % varname)

        self.hw_need["Regs"] += 1
        return self

    def freeze(self):
        #bw_req = self.read_footprint/self.compute_cycles
        self.mem_cycles = self.read_footprint/self.max_bw
        self._freeze = True
        self.mem_cycles *= self.scale_cycle;
        self.compute_cycles *= self.scale_cycle
        for k in self.hw_need.keys():
            self.hw_need[k] *= self.scale_hardware



def schedule(hwmodel: HardwareModel, expr: ast.AST):
    """[Schedules the expr from AST]

    Args:
        expr (): 
        type (): 

    Returns:
        : 
    """

    # rescheduleNodesWhenNeeded : (ALAP) rescheduling for non-memory, non-control nodes.
    # upsamplelloops
    # run
    blk = ExecutionBlock(hwmodel.max_bw)

    for op_name,op in ASTUtils.valid_operators(expr):
        # add latency and num hw ops 
        blk.add_op(hwmodel,op)

    # load variables from memory over the execution block's clock cycles
    # only need to load variables once
    variables = list(ASTUtils.get_vars(expr))
    for var in ASTUtils.get_vars(expr):
        blk.load_from_memory(hwmodel,var)

    blk.freeze()
    return blk



def process_expr(hwmodel: HardwareModel, expr: ast.AST, loop_variables=[]):
    """[Parse the input Python Code file]

    Args:
        expr ():
        type ():
        unrolled (int, optional): . Defaults to 1.
        loop_iters (int, optional): . Defaults to 1.
    """
    execution_block : ExecutionBlock = schedule(hwmodel,expr)

    if execution_block is None:
        return

    #global cycles, hw_allocated, hw_utilized

    if len(loop_variables) == 0:
        hwmodel.add_execution_block(expr,execution_block)
    else:
        unrolled_block = execution_block.copy()
        loop_iters = list(map(lambda lv: hwmodel.get_loop_count(lv),loop_variables))
        unroll_iters = list(map(lambda lv: hwmodel.get_unroll_count(lv),loop_variables))

        print(loop_iters)
        print(unroll_iters)
        for li,ui in zip(loop_iters,unroll_iters):
            unrolled_block.scale_cycles(li/ui)
            unrolled_block.scale_hardware_elements(ui)
        hwmodel.add_execution_block(expr,unrolled_block)



def process_for_stmt(model: HardwareModel, node: ast.AST, loop_variables=[]):
    assert(isinstance(node, ast.For))
    varname = ASTUtils.get_identifier(node)
    for stmt in node.body:
        process_stmt(model, stmt, loop_variables=[varname] + loop_variables)


def process_stmt(model: HardwareModel, node: ast.AST, loop_variables=[]):
    """

    Args:
        string (): 
        unrolled (int, optional): . Defaults to 1.
        loop_iters (int, optional): . Defaults to 1.
    """
    kwargs = {"loop_variables":loop_variables}
    if type(node) == ast.Call or \
       type(node) == ast.Import or \
       type(node) == ast.Name:
        return None

    elif type(node) == ast.If:
        process_stmt(model, node.test,  **kwargs)

    elif type(node) == ast.Return:
        process_stmt(model, node.value, **kwargs)


    elif type(node) == ast.FunctionDef:
        for stmt in node.body:
            process_stmt(model, stmt, **kwargs)

    elif type(node) == ast.Assign or \
         type(node) == ast.AugAssign or \
         type(node) == ast.Expr or \
         type(node) == ast.BinOp or \
         type(node) == ast.Compare or \
         type(node) == ast.BoolOp:
         # allocated memory/regnodesters
         process_expr(model, node, **kwargs)

    elif type(node) == ast.For:
        process_for_stmt(model,node,**kwargs)

    elif type(node) == ast.Tuple:
        for elt in node.elts:
            process_stmt(model,elt,**kwargs)

    else:
        #parse_code(model, node,  unrolled=unrolled, loop_iters=loop_iters)
        raise Exception("stmt unhandled <%s> %s" % (node, astor.to_source(node)))

def astpr(a:ast.AST):
    #return str(ast.dump(a))
    return str(astor.to_source(a))

def load_variables(model: HardwareModel, graph: cfg_model.CFG):
    for node in graph:
        for i in node.statements:
            if type(i) == ast.Assign:
                # allocated memory/registers
                srcline = astor.to_source(i.value)
                # original implementation only resolves lists of constants
                nbytes = ASTUtils.dynamic_get_size(srcline)

                for targ in i.targets:
                    for name in list(ASTUtils.get_vars(targ)):
                        model.declare_global_var(name,nbytes)
                        print("DECL <%s> size=%d" % (name,nbytes))



def map_cfg_to_execution_blocks(graph : cfg_model.CFG, loop_counts={}, var_sizes={}, given_bandwidth=1000000):
    """
    Parse a non-AI workload graph and store the configuration as a hardware representation 
    """
    model = HardwareModel(bandwidth=given_bandwidth, \
                          loop_counts=loop_counts, \
                          var_sizes=var_sizes)
    unroll_params = {}

    #print("===== Loading Global Variables ====")
    #load_variables(model, graph)

    print("===== Loading Global Variables ====")
    for node in graph:
        # yield(node)
        print("\n==== NODE %s ====" % node)
        for i in node.statements:
            process_stmt(model,i)


    return model
