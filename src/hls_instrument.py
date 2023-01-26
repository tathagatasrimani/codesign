import ast
import astor

from cfg.ast_utils import ASTUtils
import os

def _parse(fmtstr,**kwargs):
    return ast.parse(fmtstr.format(**kwargs))

class ForLoopRewriter(ast.NodeTransformer):

    def __init__(self):
        super().__init__()


    @staticmethod
    def mkblock(stmts):
        return ast.Module(stmts)

    def visit_Assign(self, node):
        sizes = []
        for targ in node.targets:
            varname = astor.to_source(targ).strip()
            lvname = ASTUtils.get_identifier(node)

            log_stmt = _parse("print(\"@DRAGON VARSIZE {lvname} {name} %d\" % (sys.getsizeof({expr})))",
                              lvname=lvname, name=varname, expr=varname)
            sizes.append(log_stmt)

        ast.Assign(node.targets, node.value)
        block = ForLoopRewriter.mkblock([node]+sizes)
        return block

    def visit_For(self, node):
        # redirect the assignment to a usually invalid variable name so it
        # doesn't clash with other variables in the code
        target = ast.Name('@loop_var', ast.Store())
        lvname = ASTUtils.get_identifier(node)


        value_expr = astor.to_source(node.iter)
        assign_stmt = _parse("{name} = list({expr})",
                          name=lvname, expr=value_expr)
        log_stmt = _parse("print(\"@DRAGON LOOP {name} %d\" % (len({expr})))",
                          name=lvname, expr=lvname)
        new_iter_args = ast.Name(lvname)

        target=self.generic_visit(node.target)
        new_body = map(lambda stmt: self.generic_visit(stmt), node.body)
        new_orelse = map(lambda stmt: self.generic_visit(stmt), node.orelse) if \
            len(node.orelse) > 0 else []
        new_node = ast.For(target, new_iter_args, new_body, new_orelse)
        block = ForLoopRewriter.mkblock([assign_stmt, log_stmt, new_node])
        ast.fix_missing_locations(block)
        return block

    #def generic_visit(self,node):
    #    return node

def parse_result(logname):
    def add(d,k,v):
        if not k in d:
            d[k] = []
        d[k].append(v)

    loopvars = {}
    sizevars = {}
    with open(logname, 'r') as fh:
        for line in fh:
            args = line.strip().split(" ")
            assert(args[0] == "@DRAGON")
            if args[1] == "LOOP":
                add(loopvars, args[2], int(args[3]))


            elif args[1] == "VARSIZE":
                add(sizevars, args[2], (args[3],int(args[-1])))

    return loopvars, sizevars

def instrument_and_run(filepath:str):
    with open(filepath, 'r') as src_file:
        src = src_file.read()

        codeobj = compile(src, 'tmp.py', 'exec')
        tree = ast.parse(src, mode='exec')

        rewrite_tree = ForLoopRewriter().visit(tree)
        with open("tmp.py", 'w') as fh:
            fh.write("import sys\n")
            fh.write(astor.to_source(rewrite_tree))

        cmd = "python3 {file} | grep '^@DRAGON' | sort -u > {logfile}" \
            .format(file="tmp.py",logfile="log.txt")
        os.system(cmd)

        return parse_result("log.txt")

