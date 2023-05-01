from enum import Enum

class ControlFlowType(Enum):
    IfTaken = "if-taken"
    IfNotTaken = "if-nottaken"
    LoopEntered = "loop-enter"
    LoopExited = "loop-exit"
    LoopFinished = "loop-finish"
    FunctionEntered = "func-enter"
    FunctionExited = "func-exit"

class ControlFlowInfo:

    def __init__(self,typ,lineno):
        self.typ = typ
        self.lineno = lineno
        self.taken = False

    def __repr__(self):
        return "%s.%s" % (self.typ,self.lineno)


INSTR_DATA = {}

def instrument_log(text,e):
    print("[LOG] %s = %s" % (text,e))
    return e

def define_path(typstr,lineno):
    info = ControlFlowInfo(ControlFlowType(typstr),lineno)
    INSTR_DATA[(info.typ,info.lineno)] = info
    print("DEFINE %s" % info)

def visit_path(typstr,lineno):
    instr = INSTR_DATA[(ControlFlowType,lineno)]
    instr.taken = True
    print("TAKE %s" % instr)
