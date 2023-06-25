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
            return "Mult" # may change this later
        elif type(op) == ast.Div  or type(op) == ast.FloorDiv:
            return "FloorDiv"
        elif type(op) == ast.Mod:
            return "Mod"
        elif type(op) == ast.LShift:
            return "LShift"
        elif type(op) == ast.RShift:
            return "RShift"
        elif type(op) == ast.BitOr:
            return "BitOr"
        elif type(op) == ast.BitAnd:
            return "BitAnd"
        elif type(op) == ast.BitXor:
            return "BitXor"
        else:
            raise Exception("unhandled binary operator <%s>" % op)



    @staticmethod
    def unaryop_to_opname(op:ast.unaryop):
        if type(op) == ast.Invert:
            return "Invert"
        elif type(op) == ast.USub:
            return "USub"
        elif type(op) == ast.Not:
            return "Not"
        elif type(op) == ast.UAdd:
            return "UAdd"
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
        elif type(op) == ast.Is:
            return None
        elif type(op) == ast.IsNot:
            return None
        elif type(op) == ast.In:
            return "Eq"
        elif type(op) == ast.NotIn:
            return None
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
            return None

        elif isinstance(expr, ast.UnaryOp):
            return None

        elif type(expr) == ast.BoolOp:
            return None

        elif type(expr) == ast.Compare:
            return None
                
        elif type(expr) == ast.AugAssign:
            return None 
 
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
            print("unhandled expression <%s>" % expr)
            return None
    
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
        #Note: should I have the orelse component for the ast.for, ast.while, and ast.if objects? Was thinking no.
        elif type(expr) == ast.For or type(expr) == ast.AsyncFor:
            return [expr.target, expr.iter]
        elif type(expr) == ast.While:
            return [expr.test]
        elif type(expr) == ast.If:
            return [expr.test]
        elif type(expr) == ast.With or type(expr) == ast.AsyncWith:
            return expr.items
        elif type(expr) == ast.Match:
            return [expr.subject] + expr.cases
        elif type(expr) == ast.Raise:
            return [expr.exc, expr.cause]
        elif type(expr) == ast.Compare:
            return [expr.left] + expr.ops + expr.comparators
        elif type(expr) == ast.Expr:
            return [expr.value]
        else:
            if expr: print("unhandled (sub) expresssion <%s>" % expr)
            return []
        
    # modules #
    ############### 
    @staticmethod
    def isModule(mod):
        return type(mod) == ast.Module
    
    @staticmethod
    def isInteractive(mod):
        return type(mod) == ast.Interactive
    
    @staticmethod
    def isExpression(mod):
        return type(mod) == ast.Expression
    
    @staticmethod
    def isFunctionType(mod):
        return type(mod) == ast.FunctionType
    
    # statements #
    ############### 
    @staticmethod
    def isFunctionDef(stmt):
        return type(stmt) == ast.FunctionDef
    
    @staticmethod
    def isAsyncFunctionDef(stmt):
        return type(stmt) == ast.AsyncFunctionDef
    
    @staticmethod
    def isClassDef(stmt):
        return type(stmt) == ast.ClassDef
    
    @staticmethod
    def isReturn(stmt):
        return type(stmt) == ast.Return
    
    @staticmethod
    def isDelete(stmt):
        return type(stmt) == ast.Delete
    
    @staticmethod
    def isAssign(stmt):
        return type(stmt) == ast.Assign
    
    @staticmethod
    def isAugAssign(stmt):
        return type(stmt) == ast.AugAssign
    
    @staticmethod
    def isAnnAssign(stmt):
        return type(stmt) == ast.AnnAssign
    
    @staticmethod
    def isFor(stmt):
        return type(stmt) == ast.For
    
    @staticmethod
    def isAsyncFor(stmt):
        return type(stmt) == ast.AsyncFor
    
    @staticmethod
    def isWhile(stmt):
        return type(stmt) == ast.While
    
    @staticmethod
    def isIf(stmt):
        return type(stmt) == ast.If
    
    @staticmethod
    def isWith(stmt):
        return type(stmt) == ast.With
    
    @staticmethod
    def isAsyncWith(stmt):
        return type(stmt) == ast.AsyncWith
    
    @staticmethod
    def isMatch(stmt):
        return type(stmt) == ast.Match
    
    @staticmethod
    def isRaise(stmt):
        return type(stmt) == ast.Raise
    
    @staticmethod
    def isTry(stmt):
        return type(stmt) == ast.Try
    
    @staticmethod
    def isTryStar(stmt):
        return type(stmt) == ast.TryStar
    
    @staticmethod
    def isAssert(stmt):
        return type(stmt) == ast.Assert
    
    @staticmethod
    def isImport(stmt):
        return type(stmt) == ast.Import
    
    @staticmethod
    def isImportFrom(stmt):
        return type(stmt) == ast.ImportFrom
    
    @staticmethod
    def isGlobal(stmt):
        return type(stmt) == ast.Global
    
    @staticmethod
    def isNonlocal(stmt):
        return type(stmt) == ast.Nonlocal
    
    @staticmethod
    def isExpr(stmt):
        return type(stmt) == ast.Expr

    # expressions #
    ###############        
    @staticmethod
    def isBoolOp(expr):
        return type(expr) == ast.BoolOp
    
    @staticmethod
    def isNamedExpr(expr):
        return type(expr) == ast.NamedExpr
    
    @staticmethod
    def isBinOp(expr):
        return type(expr) == ast.BinOp
    
    @staticmethod
    def isUnaryOp(expr):
        return type(expr) == ast.UnaryOp
    
    @staticmethod
    def isLambda(expr):
        return type(expr) == ast.Lambda
    
    @staticmethod
    def isIfExp(expr):
        return type(expr) == ast.IfExp
    
    @staticmethod
    def isDict(expr):
        return type(expr) == ast.Dict
    
    def isSet(expr):
        return type(expr) == ast.Set
    
    @staticmethod
    def isListComp(expr):
        return type(expr) == ast.ListComp
    
    @staticmethod
    def isSetComp(expr):
        return type(expr) == ast.SetComp
    
    @staticmethod
    def isDictComp(expr):
        return type(expr) == ast.DictComp
    
    @staticmethod
    def isGeneratorExp(expr):
        return type(expr) == ast.GeneratorExp
    
    @staticmethod
    def isAwait(expr):
        return type(expr) == ast.Await
    
    @staticmethod
    def isYield(expr):
        return type(expr) == ast.Yield
    
    @staticmethod
    def isYieldFrom(expr):
        return type(expr) == ast.YieldFrom
    
    @staticmethod
    def isCompare(expr):
        return type(expr) == ast.Compare
    
    @staticmethod
    def isCall(expr):
        return type(expr) == ast.Call
    
    @staticmethod
    def isFormattedValue(expr):
        return type(expr) == ast.FormattedValue
    
    @staticmethod
    def isJoinedStr(expr):
        return type(expr) == ast.JoinedStr
    
    @staticmethod
    def isConstant(expr):
        return type(expr) == ast.Constant
    
    @staticmethod
    def isAttribute(expr):
        return type(expr) == ast.Attribute
    
    @staticmethod
    def isSubscript(expr):
        return type(expr) == ast.Subscript
    
    @staticmethod
    def isStarred(expr):
        return type(expr) == ast.Starred
    
    @staticmethod
    def isName(expr):
        return type(expr) == ast.Name
    
    @staticmethod
    def isList(expr):
        return type(expr) == ast.List
    
    @staticmethod
    def isTuple(expr):
        return type(expr) == ast.Tuple
    
    @staticmethod
    def isSlice(expr):
        return type(expr) == ast.Slice

