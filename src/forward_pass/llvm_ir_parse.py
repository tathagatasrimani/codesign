def store_op(instruction):
    _, _, op, _, src, _, dst = instruction.split()
    src = [src.strip(",")]
    dst = dst.strip(",")
    return {"op": op, "src": src, "dst": dst}

def write_op(instruction):
    #print(f"write_op: {instruction}")
    if len(instruction.split()) == 9:
        _, _, op, _, _, _, src, _, dst = instruction.split()
    else:
        _, _, op, _, _, _, src, _, dst, _, _ = instruction.split()
    src = [src.strip(",")]
    dst = dst.strip(",")
    return {"op": op, "src": src, "dst": dst}

def read_op(instruction):
    dst, _, op, _, _, _, src = instruction.split()
    src = [src.strip(",")]
    return {"op": op, "src": src, "dst": dst}

def request_op(instruction):
    dst, _, op, _, _, _, src, _, _ = instruction.split()
    src = [src.strip(",")]
    return {"op": op, "src": src, "dst": dst}

def load_op(instruction):
    dst, _, op, _, src = instruction.split()
    src = [src.strip(",")]
    return {"op": op, "src": src, "dst": dst}

def arith_op(instruction):
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
    src = [src.strip(",")]
    return {"op": op, "src": src, "dst": dst}

def undef_num_src_op(instruction):
    dst, _, op, _, _, _ = instruction.split()[0:6]
    # not set amount of srcs, every other element is a src, others are formats
    src = instruction.split()[6::2]
    for i in range(len(src)):
        src[i] = src[i].strip(",")
    return {"op": op, "src": src, "dst": dst}

def undef_num_src_op_all_srcs(instruction):
    dst, _, op = instruction.split()[0:3]
    src = instruction.split()[4::2]
    for i in range(len(src)):
        src[i] = src[i].strip(",")
    return {"op": op, "src": src, "dst": dst}

def parse_op(instruction, op_name):
    if op_name == "store":
        parsed_op = store_op(instruction)
    elif op_name == "read":
        parsed_op = read_op(instruction)
    elif op_name == "load":
        parsed_op = load_op(instruction)
    elif op_name == "icmp":
        parsed_op = arith_op(instruction)
    elif op_name == "add":
        parsed_op = arith_op(instruction)
    elif op_name == "sub":
        parsed_op = arith_op(instruction)
    elif op_name == "mul":
        parsed_op = arith_op(instruction)
    elif op_name == "div":
        parsed_op = arith_op(instruction)
    elif op_name == "select":
        parsed_op = src_3_op(instruction)
    elif op_name == "zext":
        parsed_op = unary_op(instruction)
    elif op_name == "getelementptr":
        parsed_op = undef_num_src_op_all_srcs(instruction)
    elif op_name == "shl":
        parsed_op = arith_op(instruction)
    elif op_name == "bitcast":
        parsed_op = unary_op(instruction)
    elif op_name == "fmul":
        parsed_op = arith_op(instruction)
    elif op_name == "fadd":
        parsed_op = arith_op(instruction)
    elif op_name == "fdiv":
        parsed_op = arith_op(instruction)
    elif op_name == "call":
        parsed_op = undef_num_src_op_all_srcs(instruction)
    elif op_name == "partselect":
        parsed_op = src_4_op(instruction)
    elif op_name == "urem":
        parsed_op = arith_op(instruction)
    elif op_name == "or":
        parsed_op = arith_op(instruction)
    elif op_name == "bitconcatenate":
        parsed_op = undef_num_src_op(instruction)
    elif op_name == "mux":
        parsed_op = undef_num_src_op(instruction)
    elif op_name == "trunc":
        parsed_op = unary_op(instruction)
    elif op_name == "insertvalue":
        parsed_op = arith_op(instruction)
    elif op_name == "bitselect":
        parsed_op = undef_num_src_op(instruction)
    elif op_name == "xor":
        parsed_op = arith_op(instruction)
    elif op_name == "sext":
        parsed_op = unary_op(instruction)
    elif op_name == "and":
        parsed_op = arith_op(instruction)
    elif op_name == "phi":
        parsed_op = undef_num_src_op_all_srcs(instruction) # check this
    elif op_name == "fcmp":
        parsed_op = arith_op(instruction)
    elif op_name == "extractvalue":
        parsed_op = unary_op(instruction)
    elif op_name == "write":
        parsed_op = write_op(instruction)
    elif op_name == "writereq":
        parsed_op = request_op(instruction)
    elif op_name == "writeresp":
        parsed_op = read_op(instruction)
    elif op_name == "readreq":
        parsed_op = request_op(instruction)
    elif op_name == "readresp":
        parsed_op = read_op(instruction)
    elif op_name == "lshr":
        parsed_op = arith_op(instruction)
    elif op_name == "switch":
        parsed_op = undef_num_src_op_all_srcs(instruction)
    else:
        raise ValueError(f"Unexpected op name: {op_name} for instruction: {instruction}")
    parsed_op["type"] = "op" if op_name != "call" else "serial"
    parsed_op["call_function"] = "N/A" if op_name != "call" else parsed_op["src"][0].strip(",").strip("@")
    return parsed_op
    
