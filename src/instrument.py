from _ast import Attribute, AugAssign, Subscript, arguments, keyword
import ast
from typing import Any
import astor
import sys
from staticfg.builder import CFGBuilder
from ast_utils import ASTUtils
from collections import deque
import copy

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

class NameScopeInstrumentor(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.scope = deque([0])
        self.next_scope = 1
        self.var_scopes = {}
        self.valid_scopes = set()
        self.dont_change_names = set(("__name__", "__main__", "loop", "range", "self", "time", "np", "int", "str", "math", "heapdict"))
    
    def get_name(self, name, scope):
        return name + "__" + scope

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

    def visit_For(self, node):
        report("visiting for", node)
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        scope = self.enter_and_exits()
        cur_scope = self.scope[-1]
        self.valid_scopes.add(cur_scope)
        node.body = self.visit_Stmts(node.body)
        node.body = [scope[0]] + node.body + [scope[1]]
        node.orelse = self.visit_Stmts(node.orelse)
        if cur_scope in self.valid_scopes: 
            self.valid_scopes.remove(cur_scope)
        self.clean_up_scope(cur_scope)
        return node
    
    def visit_AugAssign(self, node: AugAssign) -> Any:
        if node.lineno not in lineno_to_node: return node
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        stmt = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')\n')
        return ProgramInstrumentor.mkblock([stmt, node])
    
    def visit_Assign(self, node):
        if node.lineno not in lineno_to_node: return node
        node.targets = self.visit_Stmts(node.targets)
        node.value = self.visit(node.value)
        stmt = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')\n')
        return ProgramInstrumentor.mkblock([stmt, node])

    def visit_Call(self,node):
        report("visiting function call",node)
        if node.args: node.args = self.visit_Stmts(node.args)
        if node.keywords: node.keywords = self.visit_Stmts(node.keywords)
        if type(node.func) == ast.Attribute: node.func = self.visit(node.func)
        return ProgramInstrumentor.mkblock([node])

    def visit_FunctionDef(self,node):
        if node.lineno not in lineno_to_node: return node
        self.dont_change_names.add(node.name)

        stmt1 = text_to_ast('print(' + str(lineno_to_node[node.lineno]) + ',' + str(node.lineno) + ')')
        print("visiting function def", node)
        scope = self.enter_and_exits()
        cur_scope = self.scope[-1]
        self.valid_scopes.add(cur_scope)
        stmts = []
        for arg in node.args.args:
            name = ast.Name(id=arg.arg, ctx=ast.Store())
            name = self.visit(name)
            stmts.append(ast.Assign(targets=[name], value=ast.Name(id=arg, ctx=ast.Load())))
        node.body = self.visit_Stmts(node.body)
        if cur_scope in self.valid_scopes: self.valid_scopes.remove(cur_scope)
        self.clean_up_scope(cur_scope)
        report("visiting func def",node)
        new_body = [scope[0], stmt1] + stmts + node.body + [scope[1]]
        return ast.FunctionDef(node.name,args=node.args, body=new_body, \
                               decorator_list=node.decorator_list, lineno=node.lineno)
    
    def visit_arguments(self, node: arguments) -> Any:
        if node.posonlyargs: node.posonlyargs = self.visit_Stmts(node.posonlyargs)
        if node.args: node.args = self.visit_Stmts(node.args)
        if node.kwonlyargs: node.kwonlyargs = self.visit_Stmts(node.kwonlyargs)
        if node.defaults: node.defaults = self.visit_Stmts(node.defaults)
        if node.vararg: node.vararg = self.visit(node.vararg)
        if node.kwarg: node.kwarg = self.visit(node.kwarg)
        if node.kw_defaults: node.kw_defaults = self.visit_Stmts(node.kw_defaults)
        report("visiting arguments", node)
        return node
    
    def visit_arg(self, node):
        report("visiting arg", node)
        if node.arg not in self.var_scopes or len(self.var_scopes[node.arg]) == 0:
            self.var_scopes[node.arg] = deque([self.scope[-1]])
        if node.arg not in self.dont_change_names: node.arg = self.get_name(node.arg, str(self.var_scopes[node.arg][-1]))
        return node
    
    def visit_Return(self, node):
        if node.value: node.value = self.visit(node.value)
        stmts = []
        for scope in self.valid_scopes:
            stmts.append(text_to_ast("print(\'exit scope " + str(scope) + "\')"))
        return ProgramInstrumentor.mkblock(stmts + [node])

    def visit_Name(self, node):
        report("visiting name", node)
        if node.id in self.dont_change_names: return node
        if (type(node.ctx) == ast.Store):
            if node.id not in self.var_scopes:
                self.var_scopes[node.id] = deque([self.scope[-1]])
                return ast.Name(id=self.get_name(node.id, str(self.scope[-1])), ctx=node.ctx)
            else:
                while len(self.var_scopes[node.id]) > 0 and self.var_scopes[node.id][-1] not in self.valid_scopes:
                    self.var_scopes[node.id].pop()
                if len(self.var_scopes[node.id]) > 0:
                    return ast.Name(id=self.get_name(node.id, str(self.var_scopes[node.id][-1])), ctx=node.ctx)
                else:
                    self.var_scopes[node.id] = deque([self.scope[-1]])
                    return ast.Name(id=self.get_name(node.id, str(self.scope[-1])), ctx=node.ctx)
        else:
            if node.id not in self.var_scopes or len(self.var_scopes[node.id]) == 0:
                self.var_scopes[node.id] = deque([self.scope[-1]])
            return ast.Name(id=self.get_name(node.id, str(self.var_scopes[node.id][-1])), ctx=node.ctx)

# this class transforms a program AST and injects commands
class ProgramInstrumentor(ast.NodeTransformer):

    def __init__(self):
        super().__init__()
        self.preamble = []
        
    def add_preamble(self,stmt):
        self.preamble.append(stmt)

    def get_name(self, name, scope):
        return name + "__" + scope

    @staticmethod
    def mkblock(stmts):
        block = ast.Module(stmts)
        ast.fix_missing_locations(block)
        return block

    def visit_Stmts(self,stmts):
        return list(map(lambda stmt: self.visit(stmt), stmts))
    
    def name_extras(self, node, var_name):
        stmt1 = text_to_ast('print(\'malloc\', sys.getsizeof(' + var_name + '), \'' + node.id + '\')')
        new_line = ast.Name(node.id, ctx=ast.Load())
        return [self.visit(new_line), stmt1]
    
    # so far only covering targets[0]
    def visit_Assign(self,node):
        node.targets = self.visit_Stmts(node.targets)
        node.value = self.visit(node.value)
        report("visiting assignment",node)
        block = []
        if type(node.targets[0]) == ast.Name:
            block = [node, text_to_ast('write_')] + self.name_extras(node.targets[0], node.targets[0].id)
        elif type(node.targets[0]) == ast.Tuple:
            for var in node.targets[0].elts:
                if type(var) == ast.Name:
                    block = [node, text_to_ast('write_')] + self.name_extras(var, var.id)
        elif type(node.targets[0]) == ast.Subscript:
            new_line = ast.Subscript(node.targets[0].value, node.targets[0].slice, ctx=ast.Load())
            block = [node, text_to_ast('write_'),self.visit(new_line)]
        else:
            block = [node]
        return ProgramInstrumentor.mkblock(block)
    
    def visit_AugAssign(self, node: AugAssign):
        node.target = self.visit(node.target)
        node.op = self.visit(node.op)
        node.value = self.visit(node.value)
        report("visiting augmented assignment",node)
        block = []
        if type(node.target) == ast.Name:
            new_line = ast.Name(node.target.id, ctx=ast.Load())
            block = [node, text_to_ast('write_'), self.visit(new_line)]
        elif type(node.target) == ast.Subscript:
            val = copy.deepcopy(node.target.value)
            slice = copy.deepcopy(node.target.slice)
            new_line = ast.Subscript(val, slice, ctx=ast.Load())
            new_node = ast.AugAssign(node.target, node.op, node.value)
            print(ast_to_text(new_node))
            block = [new_node, text_to_ast('write_'), self.visit(new_line)]
            print(ast_to_text(new_node))
            print(ast_to_text(ProgramInstrumentor.mkblock(block)))
        else:
            print("hello")
            block = [node]
        return ProgramInstrumentor.mkblock(block)

    def visit_Call(self,node):
        report("visiting function call",node)
        if type(node.func) == ast.Name and node.func.id == "print": return node
        node.args = self.visit_Stmts(node.args)
        if type(node.func) == ast.Attribute: node.func = self.visit(node.func)
        return ProgramInstrumentor.mkblock([node])

    def visit_Name(self, node):
        report("visiting name", node)
        if (type(node.ctx) == ast.Load):
            return ast.Call(ast.Name('instrument_read', ast.Load()), args=[ 
                                node, 
                                ast.Constant(node.id)
                            ], keywords=[])
        return node

    def visit_Subscript(self, node: Subscript) -> Any:
        report("visiting subscript", node)
        if (type(node.ctx) == ast.Store):
            node.value.ctx = ast.Store()
            node.value = self.visit(node.value)
            node.slice = self.visit(node.slice)
            return node
        else:
            t = ast_to_text(node.value)
            node.value = self.visit(node.value)
            node.slice = self.visit(node.slice)
            lower, upper, is_slice = "None", "None", "False"
            if type(node.slice) == ast.Slice:
                if node.slice.lower: 
                    lower = ast_to_text(node.slice.lower)[1:-1]
                if node.slice.upper: 
                    upper = ast_to_text(node.slice.upper)[1:-1]
                is_slice = "True"
                node.slice = ast.Name(id="None", ctx=ast.Load())
            retval = ast.Call(ast.Name('instrument_read_sub', ast.Load()), args=[node.value, ast.Constant(t), node.slice, ast.Name(id=lower, ctx=ast.Load()), ast.Name(id=upper, ctx=ast.Load()), ast.Name(is_slice, ast.Load())], keywords=[])
            print(ast_to_text(retval))
            return retval
        
    def visit_Attribute(self, node: Attribute) -> Any:
        report("visiting attribute", node)
        node.value = self.visit(node.value)
        return node
    

def instrument_and_run(filepath:str):
    global cfg
    with open(filepath, 'r') as src_file:
        src = src_file.read()
        cfg = CFGBuilder().build_from_file('main.c', filepath)
        for node in cfg:
            for statement in node.statements:
                lineno_to_node[statement.lineno] = node.id
        tree = ast.parse(src, mode='exec')
        names = filepath.split('/')
        dest_filepath = "instrumented_files/xformed-%s" % names[-1]
        test_instr = NameScopeInstrumentor()
        first_tree = test_instr.visit(tree)
        with open("instrumented_files/xformedpre-%s" % names[-1], 'w') as fh:
            fh.write("import sys\n")
            fh.write("from instrument_lib import *\n")
            fh.write(astor.to_source(first_tree))
            #inject print statement for total memory size"""
        with open("instrumented_files/xformedpre-%s" % names[-1], 'r') as new_src:
            src = new_src.read()
            tree = ast.parse(src, mode='exec')
            instr = ProgramInstrumentor()
            rewrite_tree = instr.visit(tree)
            with open(dest_filepath, 'w') as f:
                f.write("import sys\n")
                f.write("from instrument_lib import *\n")
                for stmt in instr.preamble:
                    f.write(ast_to_text(stmt)+"\n")
                f.write(astor.to_source(rewrite_tree))
                #inject print statement for total memory size"""


instrument_and_run(sys.argv[1])
print(lineno_to_node)
