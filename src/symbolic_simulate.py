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
    return 0.5 * (a + b + sp.Abs(a - b))

class SymbolicSimulator(AbstractSimulator):
    def __init__(self):
        self.total_passive_energy = 0
        self.execution_time = 0
        self.total_active_energy = 0
        self.cacti_exprs = {}
    
    def calculate_passive_energy(self, hw, total_execution_time):
        """
        Passive power is a function of purely the allocated hardware,
        not the actual computation being computed. This is calculated separately at the end.
        """
        passive_power = 0
        for node in hw.netlist.nodes:
            data = hw.netlist.nodes[node]
            if node == "end" or data["function"] == "nop": continue
            if data["function"] == "Buf" or data["function"] == "MainMem":
                rsc_name = data["library"][data["library"].find("__")+1:]
                logger.info(f"(passive energy) rsc name: {rsc_name}, data: {data['function']}")
                passive_power += hw_symbols.symbolic_power_passive[data["function"]]()[rsc_name]
            else:
                passive_power += (
                    hw_symbols.symbolic_power_passive[data["function"]]()
                )  # W
        self.total_passive_energy = passive_power * total_execution_time  # nJ

    def calculate_active_energy(self, hw, scheduled_dfg):
        # for each op, take power * latency
        self.total_active_energy = 0
        for node in scheduled_dfg:
            data = scheduled_dfg.nodes[node]
            if node == "end" or data["function"] == "nop": continue
            if data["function"] == "Buf" or data["function"] == "MainMem":
                rsc_name = data["library"][data["library"].find("__")+1:]
                logger.info(f"(active energy) rsc name: {rsc_name}, data: {data['function']}")
                logger.info(f"dict: {hw_symbols.symbolic_power_active[data['function']]()}")
                self.total_active_energy += hw_symbols.symbolic_power_active[data["function"]]()[rsc_name] * hw_symbols.symbolic_latency_wc[data["function"]]()[rsc_name]
            else:
                self.total_active_energy += hw_symbols.symbolic_power_active[data["function"]]() * hw_symbols.symbolic_latency_wc[data["function"]]()
    
    def calculate_execution_time(self, paths, scheduled_dfg):
        # take symbolic max over the critical paths
        self.execution_time = 0
        for path in paths:
            logger.info(f"adding path to execution time calculation: {path}")
            path_execution_time = 0
            for node in path[1]:
                data = scheduled_dfg.nodes[node]
                if node == "end" or data["function"] == "nop": continue
                if data["function"] == "Buf" or data["function"] == "MainMem":
                    rsc_name = data["library"][data["library"].find("__")+1:]
                    logger.info(f"(execution time) rsc name: {rsc_name}, data: {data['function']}")
                    path_execution_time += hw_symbols.symbolic_latency_wc[data["function"]]()[rsc_name]
                else:
                    path_execution_time += hw_symbols.symbolic_latency_wc[data["function"]]()
            self.execution_time = symbolic_convex_max(self.execution_time, path_execution_time).simplify() if self.execution_time != 0 else path_execution_time

        logger.info(f"symbolic execution time: {self.execution_time}")
    
    def calculate_edp(self, hw, paths, scheduled_dfg):
        """
        Calculate energy-delay product.

        Args:
            hw (object): Hardware object.
            paths (list): List of longest paths from forward pass.
            scheduled_dfg (nx.DiGraph): Scheduled data flow graph.

        Returns:
            dict: CACTI substitutions.
        """

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

        cacti_subs = {}
        
        for mem in hw.memories:
            if hw.memories[mem]["type"] == "Mem":
                MemL_expr = hw.symbolic_mem[mem].access_time * 1e9 # convert from s to ns
                MemReadEact_expr = (hw.symbolic_mem[mem].power.readOp.dynamic + hw.symbolic_mem[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                MemWriteEact_expr = (hw.symbolic_mem[mem].power.writeOp.dynamic + hw.symbolic_mem[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                MemPpass_expr = hw.symbolic_mem[mem].power.readOp.leakage # TODO: investigate units of this expr
                OffChipIOPact_expr = hw.symbolic_mem[mem].io_dynamic_power * 1e-3 # convert from mW to W

            else:
                BufL_expr = hw.symbolic_buf[mem].access_time * 1e9 # convert from s to ns
                BufReadEact_expr = (hw.symbolic_buf[mem].power.readOp.dynamic + hw.symbolic_buf[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                BufWriteEact_expr = (hw.symbolic_buf[mem].power.writeOp.dynamic + hw.symbolic_buf[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                BufPpass_expr = hw.symbolic_buf[mem].power.readOp.leakage # TODO: investigate units of this expr

            # only need to do in first iteration
            if mem not in hw_symbols.MemReadL:
                hw_symbols.MemReadL[mem] = sp.symbols(f"MemReadL_{mem}")
                hw_symbols.MemWriteL[mem] = sp.symbols(f"MemWriteL_{mem}")
                hw_symbols.MemReadEact[mem] = sp.symbols(f"MemReadEact_{mem}")
                hw_symbols.MemWriteEact[mem] = sp.symbols(f"MemWriteEact_{mem}")
                hw_symbols.MemPpass[mem] = sp.symbols(f"MemPpass_{mem}")
                hw_symbols.OffChipIOPact[mem] = sp.symbols(f"OffChipIOPact_{mem}")
                hw_symbols.BufL[mem] = sp.symbols(f"BufL_{mem}")
                hw_symbols.BufReadEact[mem] = sp.symbols(f"BufReadEact_{mem}")
                hw_symbols.BufWriteEact[mem] = sp.symbols(f"BufWriteEact_{mem}")
                hw_symbols.BufPpass[mem] = sp.symbols(f"BufPpass_{mem}")

                # update symbol table
                hw_symbols.symbol_table[f"MemReadL_{mem}"] = hw_symbols.MemReadL[mem]
                hw_symbols.symbol_table[f"MemWriteL_{mem}"] = hw_symbols.MemWriteL[mem]
                hw_symbols.symbol_table[f"MemReadEact_{mem}"] = hw_symbols.MemReadEact[mem]
                hw_symbols.symbol_table[f"MemWriteEact_{mem}"] = hw_symbols.MemWriteEact[mem]
                hw_symbols.symbol_table[f"MemPpass_{mem}"] = hw_symbols.MemPpass[mem]
                hw_symbols.symbol_table[f"OffChipIOPact_{mem}"] = hw_symbols.OffChipIOPact[mem]
                hw_symbols.symbol_table[f"BufL_{mem}"] = hw_symbols.BufL[mem]
                hw_symbols.symbol_table[f"BufReadEact_{mem}"] = hw_symbols.BufReadEact[mem]
                hw_symbols.symbol_table[f"BufWriteEact_{mem}"] = hw_symbols.BufWriteEact[mem]
                hw_symbols.symbol_table[f"BufPpass_{mem}"] = hw_symbols.BufPpass[mem]
        
            # TODO: support multiple memories in hw_symbols
            cacti_subs_new = {
                hw_symbols.MemReadL[mem]: MemL_expr,
                hw_symbols.MemWriteL[mem]: MemL_expr,
                hw_symbols.MemReadEact[mem]: MemReadEact_expr,
                hw_symbols.MemWriteEact[mem]: MemWriteEact_expr,
                hw_symbols.MemPpass[mem]: MemPpass_expr,
                hw_symbols.OffChipIOPact[mem]: OffChipIOPact_expr,

                hw_symbols.BufL[mem]: BufL_expr,
                hw_symbols.BufReadEact[mem]: BufReadEact_expr,
                hw_symbols.BufWriteEact[mem]: BufWriteEact_expr,
                hw_symbols.BufPpass[mem]: BufPpass_expr,
            }
            cacti_subs.update(cacti_subs_new)

            self.cacti_exprs[mem] = {
                "MemL_expr": MemL_expr,
                "MemReadEact_expr": MemReadEact_expr,
                "MemWriteEact_expr": MemWriteEact_expr,
                "MemPpass_expr": MemPpass_expr,
                "BufL_expr": BufL_expr,
                "BufReadEact_expr": BufReadEact_expr,
                "BufWriteEact_expr": BufWriteEact_expr,
                "BufPpass_expr": BufPpass_expr
            }

            """if not os.path.exists("src/tmp"):
                os.makedirs("src/tmp")
            with open(f"src/tmp/cacti_exprs_{mem}.txt", 'w') as f:
                txt = ""
                for expr in self.cacti_exprs[mem].keys():
                    txt += f"{expr}: {self.cacti_exprs[mem][expr]}\n"
                f.write(txt)"""

        self.calculate_execution_time(paths, scheduled_dfg)

        self.calculate_active_energy(hw, scheduled_dfg)

        self.calculate_passive_energy(hw, self.execution_time)

        self.edp = self.execution_time * (self.total_active_energy + self.total_passive_energy)

        return cacti_subs

    def save_edp_to_file(self):
        st = str(self.edp)
        
        file_path = "src/tmp/symbolic_edp.txt"
        directory = os.path.dirname(file_path)
        
        os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(st)