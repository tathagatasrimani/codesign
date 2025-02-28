import clang.cindex
import sys

# sudo apt update
# sudo apt install clang libclang-dev llvm-dev
# pip install clang
# Set the correct path to libclang (modify if necessary)


# Install instructions for ECE machines
# make sure you're on ECE filespace, not enough diskspace on andrew file system

# pip install --user clang

# mkdir -p $HOME/local
# cd $HOME/local
# wget https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/LLVM-19.1.7-Linux-X64.tar.xz
# tar -xf LLVM-19.1.7-Linux-X64.tar.xz
# mv LLVM-19.1.7-Linux-X64 llvm-19

# export PATH=$HOME/local/llvm-19/bin:$PATH
# export LD_LIBRARY_PATH=$HOME/local/llvm-19/lib:$LD_LIBRARY_PATH
# export LIBRARY_PATH=$HOME/local/llvm-19/lib:$LIBRARY_PATH
# export C_INCLUDE_PATH=$HOME/local/llvm-19/include:$C_INCLUDE_PATH
# export CPLUS_INCLUDE_PATH=$HOME/local/llvm-19/include:$CPLUS_INCLUDE_PATH

## We now have to install the right version of glibc

## we need to install texinfo first to build glibc
# mkdir -p $HOME/local
# cd $HOME/local
# wget http://ftp.gnu.org/gnu/texinfo/texinfo-6.8.tar.gz
# tar -xvzf texinfo-6.8.tar.gz
# cd texinfo-6.8
# mkdir build
# cd build
# ../configure --prefix=$HOME/local/texinfo
# make -j$(nproc)
# make install

# export PATH=$HOME/local/texinfo/bin:$PATH
# export MANPATH=$HOME/local/texinfo/share/man:$MANPATH



# mkdir -p $HOME/local/glibc
# cd $HOME/local

# wget http://ftp.gnu.org/gnu/libc/glibc-2.34.tar.gz

# tar -xvzf glibc-2.34.tar.gz
# cd glibc-2.34
# mkdir build
# cd build
# unset LD_LIBRARY_PATH
# ../configure --prefix=$HOME/local/glibc
# make -j$(nproc)
# make install

# export LD_LIBRARY_PATH=$HOME/local/glibc/lib:$LD_LIBRARY_PATH
# export LIBRARY_PATH=$HOME/local/glibc/lib:$LIBRARY_PATH
# export PATH=$HOME/local/glibc/bin:$PATH

clang.cindex.Config.set_library_file("/afs/ece.cmu.edu/user/edubbers/local/llvm-19/lib/libclang.so")

# Create Clang Index
index = clang.cindex.Index.create()

# Mapping of operators to custom C-Core functions
OPERATOR_MAP = {
    '&': 'bit_and',
    '|': 'bit_or',
    '+': 'add',
    '-': 'sub',
    '^': 'bit_xor',
    '==': 'eq',
    '!=': 'noteq',
    '<': 'lt',      # Less Than
    '<=': 'lte',    # Less Than or Equal
    '>': 'gt',      # Greater Than
    '>=': 'gte',    # Greater Than or Equal
    '~': 'not_op',
    '/': 'floordiv',
    '<<': 'left_shift',
    '>>': 'right_shift',
    '%': 'modulus',
    '*': 'multiplier'
}

# Unary operators mapping (e.g., Uadd, Usub)
UNARY_OPERATOR_MAP = {
    '-': 'sub',
    '+': 'add'
}

# Track declared variables to replace them with Register C-Core
VARIABLES_TO_REPLACE = set()

# Function to visit AST nodes and replace operators & registers
def visit_node(node, code):
    global VARIABLES_TO_REPLACE

    for child in node.get_children():
        visit_node(child, code)

    # Detect variable declarations and mark them as registers
    if node.kind == clang.cindex.CursorKind.VAR_DECL:
        var_name = node.spelling
        VARIABLES_TO_REPLACE.add(var_name)

    # Replace binary operators with C-Core function calls
    elif node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
        tokens = list(node.get_tokens())
        
        if len(tokens) >= 2:  # Ensure the node contains an operator
            left = tokens[0].spelling
            op = tokens[1].spelling
            right = tokens[2].spelling if len(tokens) > 2 else ''

            if op in OPERATOR_MAP:
                func_call = f"{OPERATOR_MAP[op]}({left}, {right})"
                start_offset = tokens[0].extent.start.offset
                end_offset = tokens[-1].extent.end.offset
                code[start_offset:end_offset] = func_call

    # Replace unary operators (Uadd, Usub)
    elif node.kind == clang.cindex.CursorKind.UNARY_OPERATOR:
        tokens = list(node.get_tokens())

        if len(tokens) >= 2:
            op = tokens[0].spelling
            operand = tokens[1].spelling

            if op in UNARY_OPERATOR_MAP:
                func_call = f"{UNARY_OPERATOR_MAP[op]}({operand})"
                start_offset = tokens[0].extent.start.offset
                end_offset = tokens[-1].extent.end.offset
                code[start_offset:end_offset] = func_call

    # Replace register accesses with C-Core read/write
    elif node.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
        var_name = node.spelling
        if var_name in VARIABLES_TO_REPLACE:
            func_call = f"{var_name}.read()"
            start_offset = node.extent.start.offset
            end_offset = node.extent.end.offset
            code[start_offset:end_offset] = func_call

    # Replace register assignments with Register C-Core write()
    elif node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
        tokens = list(node.get_tokens())
        if len(tokens) >= 3 and tokens[1].spelling == '=':  # Assignment detected
            left = tokens[0].spelling
            right = ' '.join(tok.spelling for tok in tokens[2:])  # Preserve entire right-hand expression
            if left in VARIABLES_TO_REPLACE:
                func_call = f"{left}.write({right})"
                start_offset = tokens[0].extent.start.offset
                end_offset = tokens[-1].extent.end.offset
                code[start_offset:end_offset] = func_call

# Function to process the input C file and apply transformations
def transform_code(filename):
    tu = index.parse(filename)

    with open(filename, "r") as file:
        code = list(file.read())  # Store code as a list to allow modifications

    visit_node(tu.cursor, code)

    # Save the transformed C code
    transformed_code = ''.join(code)
    with open("transformed.c", "w") as file:
        file.write(transformed_code)

    print("✅ Transformed C Code Written to 'transformed.c'")

# Run the script
if len(sys.argv) < 2:
    print("Usage: python replace_operators_registers.py <C file>")
    sys.exit(1)

transform_code(sys.argv[1])
