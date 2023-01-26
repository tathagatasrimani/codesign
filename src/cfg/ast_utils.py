import ast
import sys
import numpy as np


class ASTUtils:

    @staticmethod
    def get_identifier(expr:ast.AST):
        return "NODE_%d_%d" % (expr.lineno,expr.col_offset)


    @staticmethod
    def get_vars(expr:ast.AST):
        assert(isinstance(expr,ast.AST))
        for name in filter(lambda q: type(q) == ast.Name, ast.walk(expr)):
            yield name.id

    @staticmethod
    def valid_operators(expr:ast.AST):
        assert(isinstance(expr,ast.AST))
        for node in ast.walk(expr):
            name = ASTUtils.expr_to_opname(node)
            if not name is None:
                yield name,node
            else:
                print("[warn] expr of type <%s> not a valid operator" % type(node))

    @staticmethod
    def common_libraries():
        import time
        yield "np", sys.getsizeof(np)
        yield "time", sys.getsizeof(time)

    @staticmethod
    def dynamic_get_value(srcline:str):
        import time
        import numpy as np
        try:
            eval(srcline)
        except NameError as n:
            return None



    @staticmethod
    def dynamic_get_size(srcline:str):
        val = ASTUtils.dynamic_get_value(srcline)
        if not val is None:
            return sys.getsizeof(eval(srcline))
        else:
            return 0.0



    @staticmethod
    def operator_to_opname(op:ast.operator):
        if type(op) == ast.Add:
            return "Add"
        elif type(op) == ast.Mult:
            return "Mult"
        elif type(op) == ast.Sub:
            return "Sub"
        elif type(op) == ast.Pow:
            return None
        elif type(op) == ast.Div  or type(op) == ast.FloorDiv:
            return "FloorDiv"
        else:
            raise Exception("unhandled binary operator <%s>" % op)



    @staticmethod
    def unaryop_to_opname(op:ast.unaryop):
        if type(op) == ast.Invert:
            return "Invert"
        elif type(op) == ast.USub:
            return "USub"
        else:
            raise Exception("unhandled unary op <%s>" % op)


    @staticmethod
    def cmpop_to_opname(op:ast.cmpop):
        if type(op) == ast.Eq:
            return "Eq"
        elif type(op) == ast.LtE:
            return "LtE"
        elif type(op) == ast.Lt:
            return "Lt"
        elif type(op) == ast.GtE:
            return "GtE"
        elif type(op) == ast.Gt:
            return "Gt"


        else:
            raise Exception("unhandled comparator op <%s>" % op)


    @staticmethod
    def boolop_to_opname(expr:ast.boolop):
        if type(expr) == ast.And:
            return "And"
        elif type(expr) == ast.Or:
            return "Or"
        else:
            raise Exception("unhandled boolean operator <%s>" % expr.op)

    @staticmethod
    def expr_to_opname(expr: ast.AST):

        if isinstance(expr,ast.boolop):
            return ASTUtils.boolop_to_opname(expr)


        elif isinstance(expr,ast.unaryop):
            return ASTUtils.unaryop_to_opname(expr)


        elif isinstance(expr,ast.cmpop):
            return ASTUtils.cmpop_to_opname(expr)

        elif isinstance(expr,ast.operator):
            return ASTUtils.operator_to_opname(expr)

        elif isinstance(expr, ast.BinOp):
            return ASTUtils.operator_to_opname(expr.op)

        elif isinstance(expr, ast.UnaryOp):
            return ASTUtils.unaryop_to_opname(expr.op)

        elif type(expr) == ast.BoolOp:
            return ASTUtils.boolop_to_opname(expr.op)

        elif type(expr) == ast.Compare:
            assert(len(expr.ops) == 1)
            op = expr.ops[0]
            if type(op) == ast.UnaryOp:
                return ASTUtils.unaryop_to_opname(expr.ops[0])
            elif isinstance(op,ast.cmpop):
                return ASTUtils.cmpop_to_opname(expr.ops[0])
            else:
                raise Exception("unhandled compare operator <%s>" % expr.op)
                
        elif type(expr) == ast.AugAssign:
            return ASTUtils.operator_to_opname(expr.op) 
 
        elif type(expr) == ast.Load:
            print("[warn] ignoring loads")
            return None


        elif type(expr) == ast.Store:
            print("[warn] ignoring stores")
            return None


        elif type(expr) == ast.Return:
            return None



        elif type(expr) == ast.Assign:
            return None

        elif type(expr) == ast.arguments or \
             type(expr) == ast.arg:
            return None

        elif type(expr) == ast.Tuple:
            return None

        elif type(expr) == ast.Lambda:
            return None

        elif type(expr) == ast.Constant:
            return None
        elif type(expr) == ast.List:
            return None

        elif type(expr) == ast.Attribute:
            return None

        elif type(expr) == ast.Call:
            return None

        elif type(expr) == ast.Subscript:
            return None

        elif type(expr) == ast.Name:
            return None

        elif type(expr) == ast.keyword:
            return None  
        else:
            raise Exception("unhandled expression <%s>" % expr)
    
    @staticmethod
    def get_sub_expr(expr: ast.AST):
        if type(expr) == ast.BinOp:
            return [expr.left, expr.op, expr.right]
        elif type(expr) == ast.BoolOp:
            return expr.values + [expr.op]
        elif type(expr) == ast.NamedExpr:
            return [expr.target, expr.value]
        elif type(expr) == ast.UnaryOp:
            return [expr.op, expr.operand]
        elif type(expr) == ast.Lambda:
            return [expr.args, expr.body]
        elif type(expr) == ast.Dict:
            return expr.keys + expr.values
        elif type(expr) == ast.Set:
            return expr.elts
        elif type(expr) == ast.FunctionDef or type(expr) == ast.AsyncFunctionDef:
            return [expr.args]
        elif type(expr) == ast.ClassDef:
            return []
        elif type(expr) == ast.Return:
            return [expr.value]
        elif type(expr) == ast.Delete:
            return expr.targets
        elif type(expr) == ast.AugAssign:
            return [expr.target, expr.op, expr.value]
        elif type(expr) == ast.Assign:
            return expr.targets + [expr.value]
        elif type(expr) == ast.AnnAssign:
            return [expr.target, expr.value]
        elif type(expr) == ast.For or type(expr) == ast.AsyncFor:
            return [expr.target, expr.iter] + expr.orelse
        elif type(expr) == ast.While:
            return [expr.test] + expr.orelse
        elif type(expr) == ast.If:
            return [expr.test] + expr.orelse
        elif type(expr) == ast.With or type(expr) == ast.AsyncWith:
            return expr.items
        elif type(expr) == ast.Match:
            return [expr.subject] + expr.cases
        elif type(expr) == ast.Raise:
            return [expr.exc, expr.cause]
        else:
            #raise Exception("unhandled expresssion <%s>" % expr)
            return []

