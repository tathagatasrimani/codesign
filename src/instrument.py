from _ast import AugAssign, Subscript, arguments
import ast
from typing import Any
import astor
import sys
from staticfg.builder import CFGBuilder
from ast_utils import ASTUtils
from collections import deque

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
        self.scope = deque([0])
        self.next_scope = 1
        self.var_scopes = {}
        self.valid_scopes = set()
        self.dont_change_names = set(("__name__", "__main__", "loop", "range", "self"))

    def add_preamble(self,stmt):
        self.preamble.append(stmt)

    @staticmethod
    def mkblock(stmts):
        block = ast.Module(stmts)
        ast.fix_missing_locations(block)
        return block
    
    def clean_up_scope(self, cur_scope):
        for var in self.var_scopes:
            if len(self.var_scopes[var]) > 0 and self.var_scopes[var][-1] == cur_scope: self.var_scopes[var].pop()
        self.scope.pop()
    
    def enter_and_exits(self):
        self.scope.append(self.next_scope)
        self.next_scope += 1
        stmt = [text_to_ast("print(\'enter scope " + str(self.scope[-1]) + "\')"), text_to_ast("print(\'exit scope " + str(self.scope[-1]) + "\')")]
        return stmt

    def visit_Stmts(self,stmts):
        return list(map(lambda stmt: self.visit(stmt), stmts))
    
    def name_extras(self, node, var_name):
        stmt1 = text_to_ast('print(\'malloc\', sys.getsizeof(' + var_name + '), \'' + node.id + '\')')
        new_line = ast.Name(node.id, ctx=ast.Load())
        return [self.visit(new_line), stmt1]

    def visit_Assign(self,node):
        if node.lineno not in lineno_to_node: return node
        node.targets = self.visit_Stmts(node.targets)
        node.value = self.visit(node.value)
        report("visiting assignment",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        block = []
        if type(node.targets[0]) == ast.Name:
            block = [stmt1, node, text_to_ast('write_')] + self.name_extras(node.targets[0], node.targets[0].id)
        elif type(node.targets[0]) == ast.Tuple:
            for var in node.targets[0].elts:
                if type(var) == ast.Name:
                    block = [stmt1, node, text_to_ast('write_')] + self.name_extras(var, var.id)
        elif type(node.targets[0]) == ast.Subscript:
            new_line = ast.Subscript(node.targets[0].value, node.targets[0].slice, ctx=ast.Load())
            block = [stmt1, node, text_to_ast('write_'),self.visit(new_line)]
        else:
            block = [stmt1, node]
        return ProgramInstrumentor.mkblock(block)
    
    def visit_AugAssign(self, node: AugAssign):
        if node.lineno not in lineno_to_node: return node
        node.target = self.visit(node.target)
        node.op = self.visit(node.op)
        node.value = self.visit(node.value)
        report("visiting augmented assignment",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')\n')
        block = []
        if type(node.target) == ast.Name:
            new_line = ast.Name(node.target.id, ctx=ast.Load())
            block = [stmt1, node, text_to_ast('write_'), self.visit(new_line)]
        elif type(node.target) == ast.Subscript:
            new_line = ast.Subscript(node.target.value, node.target.slice, ctx=ast.Load())
            block = [stmt1, node, text_to_ast('write_'),self.visit(new_line)]
        else:
            print("hello")
            block = [stmt1, node]
        return ProgramInstrumentor.mkblock(block)

    def visit_If(self,node):
        if node.lineno not in lineno_to_node: return node
        scope = self.enter_and_exits()
        cur_scope = self.scope[-1]
        self.valid_scopes.add(cur_scope)
        test = self.visit(node.test)
        body = self.visit_Stmts(node.body)
        orelse = self.visit_Stmts(node.orelse)
        self.valid_scopes.remove(cur_scope)
        self.clean_up_scope(cur_scope)
        report("visiting if statement",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        stmt2 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        return ProgramInstrumentor.mkblock([scope[0], ast.If(test,[stmt1]+body,[stmt2]+orelse), scope[1]])

    def visit_For(self, node):
        report("visiting for", node)
        scope = self.enter_and_exits()
        cur_scope = self.scope[-1]
        self.valid_scopes.add(cur_scope)
        new_target = self.visit(node.target)
        report("this is the new target: ", new_target)
        self.visit(node.iter)
        self.visit_Stmts(node.body)
        self.visit_Stmts(node.orelse)
        self.valid_scopes.remove(cur_scope)
        self.clean_up_scope(cur_scope)
        new_node = ast.For(target=new_target, iter=node.iter, body=node.body, orelse=node.orelse, type_comment=node.type_comment)
        return ProgramInstrumentor.mkblock([scope[0], new_node, scope[1]])

    def visit_Call(self,node):
        report("visiting function call",node)
        return ProgramInstrumentor.mkblock([node])

    def visit_FunctionDef(self,node):
        if node.lineno not in lineno_to_node: return node
        print("visiting function def", node)
        scope = self.enter_and_exits()
        cur_scope = self.scope[-1]
        self.valid_scopes.add(cur_scope)
        new_stmts = self.visit_Stmts(node.body)
        self.visit_arguments(node.args)
        self.valid_scopes.remove(cur_scope)
        self.clean_up_scope(cur_scope)
        report("visiting func def",node)
        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        new_body = [scope[0], stmt1] + new_stmts + [scope[1]]
        return ast.FunctionDef(node.name,args=node.args, body=new_body, \
                               decorator_list=node.decorator_list)
    
    def visit_arguments(self, node: arguments) -> Any:
        for arg in node.posonlyargs: self.visit_arg(arg)
        for arg in node.args: self.visit_arg(arg)
        for arg in node.kwonlyargs: self.visit_arg(arg)
        return node
    
    def visit_arg(self, node):
        if node.arg not in self.var_scopes or len(self.var_scopes[node.arg]) == 0:
            self.var_scopes[node.arg] = deque([self.scope[-1]])
        if node.arg not in self.dont_change_names: node.arg = node.arg + str(self.var_scopes[node.arg][-1])
        return node

    def visit_Name(self, node):
        report("visiting name", node)
        if node.id in self.dont_change_names: return node
        while node.id[-1] >= '0' and node.id[-1] <= '9': node.id = node.id[:-1]
        if (type(node.ctx) == ast.Store):
            if node.id not in self.var_scopes:
                self.var_scopes[node.id] = deque([self.scope[-1]])
                return ast.Name(id=node.id + str(self.scope[-1]), ctx=node.ctx)
            else:
                while len(self.var_scopes[node.id]) > 0 and self.var_scopes[node.id][-1] not in self.valid_scopes:
                    self.var_scopes[node.id].pop()
                if len(self.var_scopes[node.id]) > 0:
                    return ast.Name(id=node.id + str(self.var_scopes[node.id][-1]), ctx=node.ctx)
                else:
                    self.var_scopes[node.id] = deque([self.scope[-1]])
                    return ast.Name(id=node.id + str(self.scope[-1]), ctx=node.ctx)
        else:
            if node.id not in self.var_scopes or len(self.var_scopes[node.id]) == 0:
                self.var_scopes[node.id] = deque([self.scope[-1]])
            node = ast.Name(id=node.id + str(self.var_scopes[node.id][-1]), ctx=node.ctx)
            return ast.Call(ast.Name('instrument_read', ast.Load()), args=[ 
                                node, 
                                ast.Constant(node.id)
                            ], keywords=[])

    def visit_Subscript(self, node: Subscript) -> Any:
        report("visiting subscript", node)
        if (type(node.ctx) == ast.Store):
            node.value.ctx = ast.Store()
            node.value = self.visit(node.value)
            node.slice = self.visit(node.slice)
            return node
        else:
            new_node = self.visit(node.value)
            new_slice = self.visit(node.slice)
            t = ast_to_text(node.value)
            retval = ast.Call(ast.Name('instrument_read_sub', ast.Load()), args=[new_node, ast.Constant(t), new_slice], keywords=[])
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
            for stmt in instr.preamble:
                fh.write(ast_to_text(stmt)+"\n")


            fh.write(astor.to_source(rewrite_tree))
            #inject print statement for total memory size


instrument_and_run(sys.argv[1])
print(lineno_to_node)
