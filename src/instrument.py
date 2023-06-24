from _ast import AugAssign
import ast
from typing import Any
import astor
import sys
from cfg.staticfg.builder import CFGBuilder

import os

lineno_to_node = {}
cfg = None

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

    def visit_Assign(self,node):
        new_node = self.generic_visit(node)
        report("visiting assignment",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        return ProgramInstrumentor.mkblock([stmt1,new_node])
    
    def visit_AugAssign(self, node: AugAssign):
        new_node = self.generic_visit(node)
        report("visiting augmented assignment",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        return ProgramInstrumentor.mkblock([stmt1,new_node])

    def visit_If(self,node):
        print(node.lineno)
        test = self.generic_visit(node.test)
        body = self.visit_Stmts(node.body)
        orelse = self.visit_Stmts(node.orelse)
        report("visiting if statement",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        stmt2 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        return ast.If(test,[stmt1]+body,[stmt2]+orelse)

    def visit_For(self,node):
        target = self.generic_visit(node.target)
        iter_ = self.generic_visit(node.iter)
        body = self.visit_Stmts(node.body)
        orelse = self.visit_Stmts(node.orelse)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        stmt2 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        forblk = ast.For(target,iter_,[stmt1]+body,orelse)
        return ProgramInstrumentor.mkblock([stmt2,forblk])


    def visit_Call(self,node):
        report("visiting function call",node)
        return ProgramInstrumentor.mkblock([node])

    def visit_FunctionDef(self,node):
        new_stmts = self.visit_Stmts(node.body)
        report("visiting func def",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        new_body = [stmt1] + new_stmts
        return ast.FunctionDef(node.name,args=node.args, body=new_body, \
                               decorator_list=node.decorator_list)


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
            for stmt in instr.preamble:
                fh.write(astor.to_source(stmt)+"\n")


            fh.write(astor.to_source(rewrite_tree))


instrument_and_run(sys.argv[1])
print(lineno_to_node)
