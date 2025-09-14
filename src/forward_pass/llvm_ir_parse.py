def debug_print(str):
    print(str)

def store_op(instruction):
    _, _, op, _, src, _, dst = instruction.split()
    src = [src.strip(",")]
    return {"op": op, "src": src, "dst": dst}

def read_op(instruction):
    dst, _, op, _, _, _, src = instruction.split()
    src = [src]
    return {"op": op, "src": src, "dst": dst}

def load_op(instruction):
    dst, _, op, _, src = instruction.split()
    src = [src]
    return {"op": op, "src": src, "dst": dst}

def arith_op(instruction):
    dst, _, op, _, src1, _, src2 = instruction.split()
    src = [src1.strip(","), src2]
    return {"op": op, "src": src, "dst": dst}

def src_2_op(instruction):
    dst, _, op, _, src1, _, src2 = instruction.split()
    src = [src1.strip(","), src2]
    return {"op": op, "src": src, "dst": dst}

def src_3_op(instruction):
    dst, _, op, _, src1, _, src2, _, src3 = instruction.split()
    src = [src1.strip(","), src2.strip(","), src3]
    return {"op": op, "src": src, "dst": dst}

def src_4_op(instruction):
    dst, _, op, _, src1, _, src2, _, src3, _, src4 = instruction.split()
    src = [src1.strip(","), src2.strip(","), src3.strip(","), src4]
    return {"op": op, "src": src, "dst": dst}

def unary_op(instruction):
    dst, _, op, _, src = instruction.split()
    src = [src]
    return {"op": op, "src": src, "dst": dst}

def undef_num_src_op(instruction):
    dst, _, op, _, _, _ = instruction.split()[0:6]
    # not set amount of srcs, every other element is a src, others are formats
    src = instruction.split()[6::2]
    return {"op": op, "src": src, "dst": dst}

def parse_op(instruction, op_name):
    debug_print(f"Instruction to parse: {instruction}")
    
    if op_name == "store":
        return store_op(instruction)
    elif op_name == "read":
        return read_op(instruction)
    elif op_name == "load":
        return load_op(instruction)
    elif op_name == "icmp":
        return arith_op(instruction)
    elif op_name == "add":
        return arith_op(instruction)
    elif op_name == "sub":
        return arith_op(instruction)
    elif op_name == "mul":
        return arith_op(instruction)
    elif op_name == "div":
        return arith_op(instruction)
    elif op_name == "select":
        return src_3_op(instruction)
    elif op_name == "zext":
        return unary_op(instruction)
    elif op_name == "getelementptr":
        return src_2_op(instruction)
    elif op_name == "shl":
        return arith_op(instruction)
    elif op_name == "bitcast":
        return unary_op(instruction)
    elif op_name == "fmul":
        return arith_op(instruction)
    elif op_name == "fadd":
        return arith_op(instruction)
    elif op_name == "call":
        return undef_num_src_op(instruction)
    elif op_name == "partselect":
        return src_4_op(instruction)
    elif op_name == "urem":
        return arith_op(instruction)
    elif op_name == "or":
        return arith_op(instruction)
    elif op_name == "bitconcatenate":
        return undef_num_src_op(instruction)
    elif op_name == "mux":
        return undef_num_src_op(instruction)
    elif op_name == "trunc":
        return unary_op(instruction)
    elif op_name == "insertvalue":
        return arith_op(instruction)
    elif op_name == "bitselect":
        return undef_num_src_op(instruction)
    elif op_name == "xor":
        return arith_op(instruction)
    elif op_name == "sext":
        return unary_op(instruction)
    elif op_name == "and":
        return arith_op(instruction)
    elif op_name == "phi":
        return src_4_op(instruction) # check this
    elif op_name == "fcmp":
        return arith_op(instruction)
    elif op_name == "extractvalue":
        return unary_op(instruction)
    elif op_name == "readreq":
        return src_3_op(instruction)
    elif op_name == "writereq":
        return src_3_op(instruction)
    elif op_name == "writeresp":
        return src_2_op(instruction)
    elif op_name == "write":
        return src_4_op(instruction)
    elif op_name == "lshr":
        return src_2_op(instruction)
    else:
        raise ValueError(f"Unexpected op name: {op_name} for instruction: {instruction}")
