import logging
import os

logger = logging.getLogger(__name__)

import sympy as sp

from .abstract_simulate import AbstractSimulator
from . import hw_symbols

def symbolic_convex_max(a, b):
    """
    Max(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b + sp.Abs(a - b, evaluate=False))

class SymbolicSimulator(AbstractSimulator):
    def __init__(self):
        self.total_passive_energy = 0
        self.execution_time = 0
        self.total_active_energy = 0
    
    def calculate_passive_energy(self, hw, total_execution_time):
        """
        Passive power is a function of purely the allocated hardware,
        not the actual computation being computed. This is calculated separately at the end.
        """
        passive_power = 0
        for node in hw.netlist.nodes:
            data = hw.netlist.nodes[node]
            passive_power += (
                hw_symbols.symbolic_power_passive[data["function"]]
            )  # W
        self.total_passive_energy = passive_power * total_execution_time  # nJ

    def calculate_active_energy(self, hw, operations):
        # for each op, take power * latency
        self.total_active_energy = 0
    
    def calculate_execution_time(self, paths):
        # take max over the critical paths
        self.execution_time = 0
    
    def calculate_edp(self, hw, paths, operations):

        # TEMPORARY SO FUNCTION COMPLETES
        MemL_expr = 0
        MemReadEact_expr = 0
        MemWriteEact_expr = 0
        MemPpass_expr = 0
        OffChipIOPact_expr = 0
        BufL_expr = 0
        BufReadEact_expr = 0
        BufWriteEact_expr = 0
        BufPpass_expr = 0
        
        for memory in hw.memories:
            if memory.type == "Mem":
                MemL_expr = hw.symbolic_mem[memory].access_time * 1e9 # convert from s to ns
                MemReadEact_expr = hw.symbolic_mem[memory].power.readOp.dynamic
                MemWriteEact_expr = hw.symbolic_mem[memory].power.writeOp.dynamic
                MemPpass_expr = hw.symbolic_mem[memory].power.readOp.leakage # TODO: investigate units of this expr
                OffChipIOPact_expr = hw.symbolic_mem[memory].io_dynamic_power * 1e-3 # convert from mW to W

            else:
                BufL_expr = hw.symbolic_buf[memory].access_time * 1e9 # convert from s to ns
                BufReadEact_expr = hw.symbolic_buf[memory].power.readOp.dynamic
                BufWriteEact_expr = hw.symbolic_buf[memory].power.writeOp.dynamic
                BufPpass_expr = hw.symbolic_buf[memory].power.readOp.leakage # TODO: investigate units of this expr

        
        # TODO: support multiple memories in hw_symbols
        cacti_subs = {
            hw_symbols.MemReadL: (MemL_expr / 2),
            hw_symbols.MemWriteL: (MemL_expr / 2),
            hw_symbols.MemReadEact: MemReadEact_expr,
            hw_symbols.MemWriteEact: MemWriteEact_expr,
            hw_symbols.MemPpass: MemPpass_expr,
            hw_symbols.OffChipIOPact: OffChipIOPact_expr,

            hw_symbols.BufL: BufL_expr,
            hw_symbols.BufReadEact: BufReadEact_expr,
            hw_symbols.BufWriteEact: BufWriteEact_expr,
            hw_symbols.BufPpass: BufPpass_expr,
        }

        self.cacti_exprs = {
            "MemL_expr": MemL_expr,
            "MemReadEact_expr": MemReadEact_expr,
            "MemWriteEact_expr": MemWriteEact_expr,
            "MemPpass_expr": MemPpass_expr,
            "BufL_expr": BufL_expr,
            "BufReadEact_expr": BufReadEact_expr,
            "BufWriteEact_expr": BufWriteEact_expr,
            "BufPpass_expr": BufPpass_expr
        }

        self.calculate_execution_time(paths)

        self.calculate_active_energy(hw, operations)

        self.calculate_passive_energy(hw, self.execution_time)

        self.edp = self.execution_time * (self.total_active_energy + self.total_passive_energy)

        if not os.path.exists("src/tmp"):
            os.makedirs("src/tmp")
        with open("src/tmp/cacti_exprs.txt", 'w') as f:
            txt = ""
            for expr in self.cacti_exprs.keys():
                txt += f"{expr}: {self.cacti_exprs[expr]}\n"
            f.write(txt)

        return cacti_subs

    def save_edp_to_file(self):
        st = str(self.edp)
        
        file_path = "src/tmp/symbolic_edp.txt"
        directory = os.path.dirname(file_path)
        
        os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(st)