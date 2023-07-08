from _ast import AugAssign, Subscript
import ast
from typing import Any
import astor
import sys
from cfg.staticfg.builder import CFGBuilder
from ast_utils import ASTUtils

import os

lineno_to_node = {}
cfg = None
last_used = {}

def text_to_ast(fmtstr,**kwargs):
    return ast.parse(fmtstr.format(**kwargs))

def ast_to_text(ast_node):
    return astor.to_source(ast_node).strip()

def get_identifier(expr:ast.AST):
    return "NODE_%d_%d" % (expr.lineno, expr.col_offset)

def unwrap_expr(node:ast.AST):
    if isinstance(node,ast.Module):
        assert(len(node.body) == 1)
        node = node.body[0]

    if not (isinstance(node,ast.Expr)):
        raise Exception("cannot unwrap, not an expression <%s>" % ast.dump(node))

    return node.value

def report(text,node):
    print("==== %s ====" % text)
    print(ast.dump(node))
    print("\n")

# this class transforms a program AST and injects commands
class ProgramInstrumentor(ast.NodeTransformer):

    def __init__(self):
        super().__init__()
        self.preamble = []

    def add_preamble(self,stmt):
        self.preamble.append(stmt)

    @staticmethod
    def mkblock(stmts):
        block = ast.Module(stmts)
        ast.fix_missing_locations(block)
        return block

    def visit_Stmts(self,stmts):
        return list(map(lambda stmt: self.visit(stmt), stmts))
    
    def name_extras(self, node, var_name):
        stmt1 = text_to_ast('print(\'malloc\', id(' + var_name + '), sys.getsizeof(' + var_name + '))')
        stmt2 = text_to_ast('memory_module.malloc(\"id(' + var_name + ')\", sys.getsizeof(' + var_name + '))')
        stmt3 = text_to_ast('print(memory_module.locations[\"id(' + var_name + ')\"].location, \"'+ var_name + '\", \"mem\")')
        new_line = ast.Name(node.id, ctx=ast.Load())
        return [self.visit(new_line), stmt1, stmt2, stmt3]

    def visit_Assign(self,node):
        if node.lineno not in lineno_to_node: return node
        new_node = self.generic_visit(node)
        report("visiting assignment",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        block = []
        if type(node.targets[0]) == ast.Name:
            block = [stmt1, new_node, text_to_ast('write_')] + self.name_extras(node.targets[0], node.targets[0].id)
        elif type(node.targets[0]) == ast.Tuple:
            for var in node.targets[0].elts:
                if type(var) == ast.Name:
                    block = [stmt1, new_node, text_to_ast('write_')] + self.name_extras(var, var.id)
        elif type(node.targets[0]) == ast.Subscript:
            new_line = ast.Subscript(node.targets[0].value, node.targets[0].slice, ctx=ast.Load())
            block = [stmt1, new_node, text_to_ast('write_'),self.visit(new_line)]
        else:
            block = [stmt1, node]
        return ProgramInstrumentor.mkblock(block)
    
    def visit_AugAssign(self, node: AugAssign):
        if node.lineno not in lineno_to_node: return node
        new_node = self.generic_visit(node)
        report("visiting augmented assignment",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')\n')
        block = []
        if type(node.target) == ast.Name:
            new_line = ast.Name(node.target.id, ctx=ast.Load())
            block = [stmt1, new_node, text_to_ast('write_'), self.visit(new_line)]
        elif type(node.target) == ast.Subscript:
            new_line = ast.Subscript(node.target.value, node.target.slice, ctx=ast.Load())
            block = [stmt1, new_node, text_to_ast('write_'),self.visit(new_line)]
        else:
            block = [stmt1, new_node]
        return ProgramInstrumentor.mkblock(block)

    def visit_If(self,node):
        if node.lineno not in lineno_to_node: return node
        test = self.generic_visit(node.test)
        body = self.visit_Stmts(node.body)
        orelse = self.visit_Stmts(node.orelse)
        report("visiting if statement",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        stmt2 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        return ast.If(test,[stmt1]+body,[stmt2]+orelse)


    def visit_Call(self,node):
        report("visiting function call",node)
        return ProgramInstrumentor.mkblock([node])

    def visit_FunctionDef(self,node):
        if node.lineno not in lineno_to_node: return node
        new_stmts = self.visit_Stmts(node.body)
        report("visiting func def",node)
        stmt1 = text_to_ast('global memory_module')
        stmt2 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        new_body = [stmt1, stmt2] + new_stmts
        return ast.FunctionDef(node.name,args=node.args, body=new_body, \
                               decorator_list=node.decorator_list)
    
    def visit_Name(self, node):
        report("visiting name", node)
        if (type(node.ctx) == ast.Store):
            return node
        else:
            return ast.Call(ast.Name('instrument_read', ast.Load()), args=[ 
                                node, 
                                ast.Constant(node.id)
                            ], keywords=[])

    def visit_Subscript(self, node: Subscript) -> Any:
        new_node = self.visit(node.value)
        report("visiting subscript", node)
        if (type(node.ctx) == ast.Store):
            return node
        else:
            t = ast_to_text(node.value)
            retval = ast.Call(ast.Name('instrument_read_sub', ast.Load()), args=[new_node, ast.Constant(t), node.slice], keywords=[])
            print(ast_to_text(retval))
            return retval
    

def instrument_and_run(filepath:str):
    global cfg
    with open(filepath, 'r') as src_file:
        src = src_file.read()
        cfg = CFGBuilder().build_from_file('main.c', filepath)
        for node in cfg:
            for statement in node.statements:
                lineno_to_node[statement.lineno] = node.id
        codeobj = compile(src, 'tmp.py', 'exec')
        tree = ast.parse(src, mode='exec')
        names = filepath.split('/')
        dest_filepath = "instrumented_files/xformed-%s" % names[-1]
        instr = ProgramInstrumentor()
        rewrite_tree = instr.visit(tree)
        with open(dest_filepath, 'w') as fh:
            fh.write("import sys\n")
            fh.write("from instrument_lib import *\n")
            fh.write("from memory import Memory\n")
            fh.write("MEMORY_SIZE = 10000\n")
            fh.write("memory_module = Memory(MEMORY_SIZE)\n")
            for stmt in instr.preamble:
                fh.write(ast_to_text(stmt)+"\n")


            fh.write(astor.to_source(rewrite_tree))
            #inject print statement for total memory size


instrument_and_run(sys.argv[1])
print(lineno_to_node)
