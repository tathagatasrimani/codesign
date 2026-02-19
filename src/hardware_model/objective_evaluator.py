"""
Pickleable ObjectiveEvaluator for multiprocessing worker functions.

This module provides a lightweight, pickleable alternative to HardwareModel
for evaluating objectives in parallel worker processes. It extracts only
the essential data needed for numeric evaluation, avoiding unpickleable
objects like cvxpy variables and sympy Relationals.
"""

import logging
import math
from typing import Dict, Any, Set

from src import sim_util
from src import coefficients

logger = logging.getLogger(__name__)

DEBUG = False
def log_info(msg):
    if DEBUG:
        logger.info(msg)

DATA_WIDTH = 16

_MEMORY_OPS = {"load", "store", "read", "write"}
_MEMORY_READ_OPS = {"load", "read"}


class ObjectiveEvaluator:
    """
    Pickleable class for evaluating objectives in worker processes.

    This class extracts only the essential data from HardwareModel needed for
    numeric objective evaluation - no cvxpy variables, no cvxpy constraints.

    The tech_model is passed directly (it's pickleable), but we avoid storing
    any HardwareModel state that contains cvxpy objects.

    Usage:
        # In main process, create from HardwareModel
        evaluator = ObjectiveEvaluator.from_hardware_model(hw)

        # Pass to worker processes (pickleable)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(worker_fn, evaluator, design_point)
                      for design_point in design_points]
    """

    def __init__(
        self,
        tech_model,
        memory_models: Dict[str, Any],
        scheduled_dfgs: Dict[str, Any],
        loop_1x_graphs: Dict[str, Any],
        dataflow_blocks: Set[str],
        netlist: Any,  # NetworkX DiGraph
        obj_fn: str,
        top_block_name: str,
        edge_to_nets: Dict[tuple, Any],
        DFF_DELAY: float,
        DFF_ENERGY: float,
        DFF_PASSIVE_POWER: float,
        DFF_AREA: float,
        mem_access_db: Dict[str, Any] = None,
    ):
        """
        Initialize the ObjectiveEvaluator.

        Args:
            tech_model: The tech model (e.g., SweepBasicModel) - pickleable
            scheduled_dfgs: Dict of scheduled DFGs (NetworkX graphs - pickleable)
            loop_1x_graphs: Dict of loop 1x graphs
            dataflow_blocks: Set of dataflow block names
            netlist: NetworkX DiGraph of the circuit netlist
            obj_fn: Objective function name ('edp', 'ed2', 'delay', 'energy')
            top_block_name: Name of the top-level block
            edge_to_nets: Dict mapping edges to nets for wire delay/energy
        """
        # Store coefficients for latency/energy calculations
        self.coeffs = coefficients.create_and_save_coefficients([7])
        self._set_coefficients()

        self.tech_model = tech_model
        self.memory_models = memory_models
        self.scheduled_dfgs = scheduled_dfgs
        self.loop_1x_graphs = loop_1x_graphs
        self.dataflow_blocks = dataflow_blocks
        self.netlist = netlist
        self.obj_fn = obj_fn
        self.top_block_name = top_block_name
        self.edge_to_nets = edge_to_nets
        self.DFF_DELAY = DFF_DELAY
        self.DFF_ENERGY = DFF_ENERGY
        self.DFF_PASSIVE_POWER = DFF_PASSIVE_POWER
        self.DFF_AREA = DFF_AREA
        # mem_access_db: mem_name -> {first_write, write_basic_block_name,
        #                              first_read,  read_basic_block_name}
        # Node labels are "<block_name>_<name_in_original_graph>".
        self.mem_access_db = mem_access_db or {}
        self.first_write_times = {}
        self.first_write_blocks = {}
        self.last_read_times = {}
        self.last_read_blocks = {}
        # Results (updated by calculate_objective)
        self.obj = 0.0
        self.execution_time = 0.0
        self.total_power = 0.0
        self.total_active_energy = 0.0
        self.total_passive_energy = 0.0
        self.total_refresh_energy = 0.0
        self.total_area = 0.0

    def _set_coefficients(self):
        """Set up logical effort coefficients."""
        self.alpha = self.coeffs["alpha"]
        self.beta = self.coeffs["beta"]
        self.gamma = self.coeffs["gamma"]
        self.area_coeffs = self.coeffs["area"]

        # TODO: add actual data for Exp16
        self.alpha["Exp16"] = 3 * (self.alpha["Mult16"] + self.alpha["Add16"])
        self.beta["Exp16"] = self.beta["Mult16"] + self.beta["Add16"]
        self.gamma["Exp16"] = 3 * (self.gamma["Mult16"] + self.gamma["Add16"])
        self.area_coeffs["Exp16"] = self.area_coeffs["Mult16"] + self.area_coeffs["Add16"]

    @classmethod
    def from_hardware_model(cls, hw) -> 'ObjectiveEvaluator':
        """
        Factory method to create an ObjectiveEvaluator from a HardwareModel.

        This extracts all necessary data and avoids cvxpy objects.

        Args:
            hw: A HardwareModel instance

        Returns:
            ObjectiveEvaluator ready for use in worker processes
        """
        return cls(
            tech_model=hw.circuit_model.tech_model,
            memory_models=hw.memory_models,
            scheduled_dfgs=hw.scheduled_dfgs,
            loop_1x_graphs=hw.loop_1x_graphs,
            dataflow_blocks=hw.dataflow_blocks,
            netlist=hw.netlist,
            obj_fn=hw.obj_fn,
            top_block_name=hw.top_block_name,
            edge_to_nets=hw.circuit_model.edge_to_nets,
            DFF_DELAY=hw.circuit_model.DFF_DELAY,
            DFF_ENERGY=hw.circuit_model.DFF_ENERGY,
            DFF_PASSIVE_POWER=hw.circuit_model.DFF_PASSIVE_POWER,
            DFF_AREA=hw.circuit_model.DFF_AREA,
            mem_access_db=hw.mem_access_db,
        )

    def set_params_from_design_point(self, design_point: Dict[str, Any]):
        """Update tech and memory models from a design point.

        Expects design_point = {"logic": {...}, "memory": {...}}.
        Falls back to treating the whole dict as logic params if "logic" key is absent.
        """
        self.tech_model.set_params_from_design_point(design_point)
        memory_config = design_point.get("memory", {})
        for mem_name, memory_model in self.memory_models.items():
            if mem_name in memory_config:
                memory_model.set_design_point(memory_config[mem_name])

    def set_clk_period(self, clk_period: float):
        """Set the clock period."""
        self.minimum_clk_period = self.calculate_minimum_clk_period()
        clk_period = max(clk_period, self.minimum_clk_period)
        self.tech_model.base_params.set_symbol_value(
            self.tech_model.base_params.clk_period, clk_period
        )

    def calculate_minimum_clk_period(self):
        #self.minimum_clk_period = sim_util.xreplace_safe(self.DFF_DELAY, self.tech_model.base_params.tech_values)
        #for edge in self.edge_to_nets:
        #    self.minimum_clk_period = max(self.minimum_clk_period, sim_util.xreplace_safe(self._wire_delay(edge) + self.DFF_DELAY, self.tech_model.base_params.tech_values))
        self.minimum_clk_period = 0
        return self.minimum_clk_period

    def calculate_objective(self):
        """
        Calculate the objective value based on execution time and energy.

        This replicates the logic from HardwareModel.calculate_objective
        but without cvxpy variables.

        Updates self.obj, self.execution_time, self.total_power,
        self.total_active_energy, and self.total_passive_energy.
        """
        self.execution_time = self.calculate_execution_time()
        self.total_passive_energy = self.calculate_passive_energy(self.execution_time)
        self.total_active_energy = self.calculate_active_energy()
        self.total_refresh_energy = self.calculate_refresh_energy()
        self.total_active_energy += self.total_refresh_energy
        self.total_power = (self.total_active_energy + self.total_passive_energy) / self.execution_time
        self.total_passive_power = self.total_passive_energy / self.execution_time
        self.total_area = self.calculate_area()

        self._save_obj_vals(self.execution_time)

    def _save_obj_vals(self, execution_time: float):
        """Calculate and save objective values based on objective function."""
        if self.obj_fn == "edp":
            self.obj = (self.total_passive_energy + self.total_active_energy) * execution_time
        elif self.obj_fn == "ed2":
            self.obj = (self.total_passive_energy + self.total_active_energy) * (execution_time ** 2)
        elif self.obj_fn == "delay":
            self.obj = execution_time
        elif self.obj_fn == "energy":
            self.obj = self.total_active_energy + self.total_passive_energy
        else:
            raise ValueError(f"Objective function {self.obj_fn} not supported")

    # =========================================================================
    # Execution Time Calculation
    # =========================================================================

    def calculate_execution_time(self) -> float:
        """Calculate execution time by traversing the scheduled DFGs."""
        self.node_arrivals = {}
        self.graph_delays = {}
        self.loop_delays_1x = {}  # loop_name -> delay of one iteration (ns)

        for basic_block_name in self.scheduled_dfgs:
            self.node_arrivals[basic_block_name] = {"full": {}, "loop_1x": {}, "loop_2x": {}}

        log_info(f"scheduled dfgs: {self.scheduled_dfgs.keys()}")

        graph_end_node = (f"graph_end_{self.top_block_name}"
                         if self.top_block_name not in self.dataflow_blocks
                         else f"{self.top_block_name}_graph_end_{self.top_block_name}")

        return self._calculate_execution_time_recursive(
            self.top_block_name,
            self.scheduled_dfgs[self.top_block_name],
            graph_end_node=graph_end_node
        )

    def _get_rsc_edge(self, edge, dfg):
        """Get the resource edge for wire delay lookup."""
        if "rsc" in dfg.nodes[edge[0]] and "rsc" in dfg.nodes[edge[1]]:
            return (dfg.nodes[edge[0]]["rsc"], dfg.nodes[edge[1]]["rsc"])
        return edge

    def _calculate_execution_time_recursive(
        self,
        basic_block_name: str,
        dfg,
        graph_end_node: str = "graph_end",
        graph_type: str = "full",
        resource_delays_only: bool = False
    ) -> float:
        """
        Recursively calculate execution time through the DFG.

        Args:
            basic_block_name: Name of the basic block
            dfg: NetworkX DiGraph representing the DFG
            graph_end_node: Name of the end node
            graph_type: Type of graph ('full', 'loop_1x', 'loop_2x')
            resource_delays_only: If True, only count resource edge delays

        Returns:
            Execution time in ns
        """
        log_info(f"calculating execution time for {basic_block_name} with graph end node {graph_end_node}")

        tv = self.tech_model.base_params.tech_values

        # Initialize all node arrivals to 0
        for node in dfg.nodes:
            self.node_arrivals[basic_block_name][graph_type][node] = 0.0

        # Process nodes
        for node in dfg.nodes:
            preds = list(dfg.predecessors(node))

            for pred in preds:
                pred_delay = 0.0

                if dfg.edges[pred, node]["resource_edge"]:
                    if dfg.nodes[pred]["function"] == "II":
                        loop_name = dfg.nodes[pred]["loop_name"]
                        delay_1x = self._calculate_execution_time_recursive(
                            basic_block_name,
                            self.loop_1x_graphs[loop_name][True],
                            graph_end_node="loop_end_1x",
                            graph_type="loop_1x",
                            resource_delays_only=True
                        )
                        self.loop_delays_1x[loop_name] = delay_1x
                        # TODO: add dependence of II on loop-carried dependency
                        pred_delay = delay_1x * (int(dfg.nodes[pred]["count"]) - 1)
                    else:
                        pred_delay = sim_util.xreplace_safe(
                            self.tech_model.base_params.clk_period, tv
                        )

                elif dfg.nodes[pred]["function"] == "Call":
                    if basic_block_name in self.dataflow_blocks:
                        pred_delay = 0.0 # ignore calls beacuse operations are inlined
                    else:
                        # Recursively calculate delay for function calls
                        call_fn = dfg.nodes[pred]["call_function"]
                        if call_fn not in self.graph_delays:
                            self.graph_delays[call_fn] = self._calculate_execution_time_recursive(
                                call_fn,
                                self.scheduled_dfgs[call_fn],
                                graph_end_node=f"graph_end_{call_fn}"
                            )
                        pred_delay = self.graph_delays[call_fn]

                elif not resource_delays_only:
                    if dfg.nodes[pred]["function"] == "Wire":
                        src = dfg.nodes[pred]["src_node"]
                        dst = dfg.nodes[pred]["dst_node"]
                        rsc_edge = self._get_rsc_edge((src, dst), dfg)
                        if rsc_edge in self.edge_to_nets:
                            pred_delay = self._wire_delay(rsc_edge)
                            log_info(f"added wire delay {pred_delay} for edge {rsc_edge}")
                        else:
                            log_info(f"no wire delay for edge {rsc_edge}")
                    else:
                        pred_delay = self._latency(dfg.nodes[pred]["function"], dfg.nodes[pred])

                log_info(f"pred_delay: {pred_delay} for node {node} and pred {pred}")

                # Update node arrival time (max of all predecessor paths)
                arrival = self.node_arrivals[basic_block_name][graph_type][pred] + pred_delay
                self.node_arrivals[basic_block_name][graph_type][node] = max(
                    arrival,
                    self.node_arrivals[basic_block_name][graph_type][node]
                )
        return self.node_arrivals[basic_block_name][graph_type][graph_end_node]

    def _wire_delay(self, edge) -> float:
        """
        Calculate wire delay for an edge.

        TODO: Replace C_diff and C_load with capacitance correctly sized
        for src and dst of each net.
        """
        tv = self.tech_model.base_params.tech_values
        wire_delay = 0.0

        for net in self.edge_to_nets[edge]:
            R_on_line = sim_util.xreplace_safe(self.tech_model.R_avg_inv, tv)
            C_current = sim_util.xreplace_safe(self.tech_model.C_diff, tv)
            wire_delay += R_on_line * C_current

            for segment in net.segments:
                layer = segment.layer
                C_layer = sim_util.xreplace_safe(
                    self.tech_model.wire_parasitics["C"][layer], tv
                )
                R_layer = sim_util.xreplace_safe(
                    self.tech_model.wire_parasitics["R"][layer], tv
                )
                C_current = segment.length * C_layer
                R_on_line += segment.length * R_layer
                wire_delay += R_on_line * C_current

            C_current = sim_util.xreplace_safe(self.tech_model.C_load, tv)
            wire_delay += R_on_line * C_current

        return wire_delay * 1e9  # Convert to ns

    def _latency(self, op_type: str, node_data: dict = None) -> float:
        """Calculate latency for an operation type.

        For memory ops (load/store/read/write) with a named memory (mem_name != 'N/A'),
        returns the cache hit or write latency from the memory model (ns).
        For register ops (mem_name == 'N/A'), returns 1 ns.
        For logic ops, uses logical effort coefficients.
        """
        if op_type in _MEMORY_OPS:
            mem_name = node_data.get("mem_name", "N/A") if node_data else "N/A"
            if mem_name != "N/A" and mem_name in self.memory_models:
                mm = self.memory_models[mem_name]
                if op_type in _MEMORY_READ_OPS:
                    return mm.cacheHitLatency_ns
                else:
                    return mm.cacheWriteLatency_ns
            else:
                op_type = "Register16"

        if op_type not in self.gamma:
            return 0.0
        tv = self.tech_model.base_params.tech_values
        delay = sim_util.xreplace_safe(self.tech_model.delay, tv)
        return math.ceil(self.gamma[op_type] * delay)

    def calculate_refresh_energy(self) -> float:
        total_refresh_energy = 0.0
        for mem_name in self.mem_access_db:
            if not hasattr(self.memory_models[mem_name], "retentionTime_ns"):
                continue
            last_read_node = self.mem_access_db[mem_name]["last_read"]
            last_read_basic_block_name = self.mem_access_db[mem_name]["last_read_basic_block_name"]
            if self.scheduled_dfgs[last_read_basic_block_name].nodes[last_read_node]["is_in_loop"]:
                loop_name = self.scheduled_dfgs[last_read_basic_block_name].nodes[last_read_node]["loop_name"]
                last_read_time = self.node_arrivals[last_read_basic_block_name]["full"][f"{last_read_basic_block_name}_loop_end_{loop_name}"] # TODO: make more precise
            else:
                last_read_time = self.node_arrivals[last_read_basic_block_name]["full"][last_read_node]
            first_write_node = self.mem_access_db[mem_name]["first_write"]
            first_write_basic_block_name = self.mem_access_db[mem_name]["write_basic_block_name"]
            first_write_time = self.node_arrivals[first_write_basic_block_name]["full"][first_write_node]
            live_time = last_read_time - first_write_time
            num_refresh = math.ceil(live_time / self.memory_models[mem_name].retentionTime_ns)
            total_refresh_energy += num_refresh * self.memory_models[mem_name].refreshEnergy_nJ
                
        return total_refresh_energy

    # =========================================================================
    # Active Energy Calculation
    # =========================================================================

    def calculate_active_energy(self) -> float:
        """
        Calculate total active energy across all basic blocks.

        Replicates HardwareModel.calculate_active_energy_vitis.
        """
        total_active_energy = 0.0
        for basic_block_name in self.scheduled_dfgs:
            total_active_energy += self._calculate_active_energy_basic_block(
                basic_block_name,
                self.scheduled_dfgs[basic_block_name]
            )
        return total_active_energy

    def _calculate_active_energy_basic_block(self, basic_block_name: str, dfg) -> float:
        """
        Calculate active energy for a single basic block.

        Replicates HardwareModel.calculate_active_energy_basic_block.
        """
        tv = self.tech_model.base_params.tech_values
        total_active_energy_basic_block = 0.0
        loop_count = 1
        loop_energy = 0.0

        for node, data in dfg.nodes(data=True):
            if data["function"] == "II":
                loop_count = int(data["count"])
                loop_name = data["loop_name"]
                loop_energy = self._calculate_active_energy_basic_block(
                    basic_block_name,
                    self.loop_1x_graphs[loop_name][False]
                )
                total_active_energy_basic_block += loop_energy * (loop_count - 1)
                log_info(f"loop count for {basic_block_name}: {loop_count}")
                log_info(f"loop energy for {basic_block_name}: {sim_util.xreplace_safe(loop_energy, tv)}")

            elif data["function"] == "Wire":
                src = data["src_node"]
                dst = data["dst_node"]
                rsc_edge = self._get_rsc_edge((src, dst), dfg)
                if rsc_edge in self.edge_to_nets:
                    total_active_energy_basic_block += self._wire_energy(rsc_edge)
                    log_info(f"edge {rsc_edge} is in edge_to_nets")
                else:
                    log_info(f"edge {rsc_edge} is not in edge_to_nets")

            else:
                energy = self._symbolic_energy_active(data["function"], data)
                total_active_energy_basic_block += energy
                log_info(f"active energy for {node}: {energy}")

        log_info(f"total active energy for {basic_block_name}: {total_active_energy_basic_block}")
        return total_active_energy_basic_block

    def _symbolic_energy_active(self, function: str, node_data: dict = None) -> float:
        """
        Calculate active energy for a function type.

        For memory ops with a named memory, returns cache hit/write dynamic
        energy from the memory model (nJ). For register ops, returns 0.
        For logic ops, uses logical effort coefficients.
        """
        if function in _MEMORY_OPS:
            mem_name = node_data.get("mem_name", "N/A") if node_data else "N/A"
            if mem_name != "N/A" and mem_name in self.memory_models:
                mm = self.memory_models[mem_name]
                if function in _MEMORY_READ_OPS:
                    return mm.cacheHitDynamicEnergy_nJ
                else:
                    return mm.cacheWriteDynamicEnergy_nJ
            else:
                function = "Register16"

        if function in ["N/A", "Call"]:
            return 0.0

        if function not in self.alpha:
            return 0.0

        tv = self.tech_model.base_params.tech_values
        alpha = self.alpha[function]
        E_act_inv = sim_util.xreplace_safe(self.tech_model.E_act_inv, tv)

        # Unpipelined energy
        unpipelined_energy = alpha * E_act_inv

        # Pipeline cost (DFF energy for additional cycles)
        DFF_ENERGY = 20 * E_act_inv  # ~20 inverters worth
        delay = sim_util.xreplace_safe(self.tech_model.delay, tv)
        clk_period = sim_util.xreplace_safe(self.tech_model.base_params.clk_period, tv)

        lat_cycles = math.ceil(self.gamma[function] * delay)
        if clk_period > 0:
            pipeline_cost = DATA_WIDTH * DFF_ENERGY * (lat_cycles / clk_period)
        else:
            pipeline_cost = 0.0

        return unpipelined_energy + pipeline_cost

    def _wire_energy(self, edge) -> float:
        """
        Calculate wire energy for an edge.

        Replicates CircuitModel.wire_energy.
        """
        tv = self.tech_model.base_params.tech_values
        V_dd = sim_util.xreplace_safe(self.tech_model.base_params.V_dd, tv)
        wire_energy = 0.0

        for net in self.edge_to_nets[edge]:
            for segment in net.segments:
                C_layer = sim_util.xreplace_safe(
                    self.tech_model.wire_parasitics["C"][segment.layer], tv
                )
                wire_energy += 0.5 * segment.length * DATA_WIDTH * C_layer * V_dd ** 2

        return wire_energy * 1e9  # Convert to nJ

    # =========================================================================
    # Passive Energy Calculation
    # =========================================================================

    def calculate_passive_energy(self, total_execution_time: float) -> float:
        """
        Calculate total passive (leakage) energy.

        Replicates HardwareModel.calculate_passive_energy_vitis.
        Memory leakage is counted once per unique memory instance.

        Args:
            total_execution_time: Execution time in ns

        Returns:
            Passive energy in nJ
        """
        total_passive_power = 0.0
        counted_memories = set()

        for node, data in self.netlist.nodes(data=True):
            power = self._symbolic_power_passive(data["function"], data, counted_memories)
            total_passive_power += power
            log_info(f"passive power for {node}: {power}")

        self.total_passive_power = total_passive_power
        return total_passive_power * total_execution_time

    def _symbolic_power_passive(self, function: str, node_data: dict = None,
                                counted_memories: set = None) -> float:
        """
        Calculate passive power for a function type.

        For memory ops with a named memory, returns cache leakage from the
        memory model (converted from mW to W). Each memory is counted only
        once via counted_memories set.
        For register ops, returns 0. For logic ops, uses logical effort coefficients.
        """
        if function in _MEMORY_OPS:
            mem_name = node_data.get("mem_name", "N/A") if node_data else "N/A"
            if mem_name != "N/A" and mem_name in self.memory_models:
                if counted_memories is not None and mem_name in counted_memories:
                    return 0.0  # already counted this memory's leakage
                if counted_memories is not None:
                    counted_memories.add(mem_name)
                return self.memory_models[mem_name].cacheLeakage_mW * 1e-3  # mW → W
            else:
                function = "Register16"

        # Unified memory/fifo nodes from physical design netlist
        if function in ("memory", "fifo"):
            mem_name = node_data.get("name", "N/A") if node_data else "N/A"
            if mem_name != "N/A" and mem_name in self.memory_models:
                if counted_memories is not None and mem_name in counted_memories:
                    return 0.0
                if counted_memories is not None:
                    counted_memories.add(mem_name)
                return self.memory_models[mem_name].cacheLeakage_mW * 1e-3  # mW → W
            return 0.0

        if function in ["N/A", "Call"]:
            return 0.0

        if function not in self.beta:
            return 0.0

        tv = self.tech_model.base_params.tech_values
        beta = self.beta[function]
        P_pass_inv = sim_util.xreplace_safe(self.tech_model.P_pass_inv, tv)

        # Unpipelined passive power
        unpipelined_power = beta * P_pass_inv

        # Pipeline cost (DFF passive power for additional cycles)
        DFF_PASSIVE_POWER = 20 * P_pass_inv  # ~20 inverters worth
        delay = sim_util.xreplace_safe(self.tech_model.delay, tv)
        clk_period = sim_util.xreplace_safe(self.tech_model.base_params.clk_period, tv)

        lat_cycles = math.ceil(self.gamma.get(function, 0) * delay)
        if clk_period > 0:
            pipeline_cost = DATA_WIDTH * DFF_PASSIVE_POWER * (lat_cycles / clk_period)
        else:
            pipeline_cost = 0.0

        return unpipelined_power + pipeline_cost

    def calculate_area(self) -> float:
        """
        Calculate area for the circuit.
        Memory area is counted once per unique memory instance.
        """
        total_area = 0.0
        counted_memories = set()
        for node, data in self.netlist.nodes(data=True):
            area = self._symbolic_area(data["function"], data, counted_memories)
            total_area += area
            log_info(f"area for {node}: {area}")
        return total_area

    def _symbolic_area(self, function: str, node_data: dict = None,
                       counted_memories: set = None) -> float:
        """
        Calculate area for a function type.

        For memory ops with a named memory, returns cache area from the
        memory model (converted from mm² to um²). Each memory is counted
        only once via counted_memories set.
        """
        if function in _MEMORY_OPS:
            mem_name = node_data.get("mem_name", "N/A") if node_data else "N/A"
            if mem_name != "N/A" and mem_name in self.memory_models:
                if counted_memories is not None and mem_name in counted_memories:
                    return 0.0  # already counted this memory's area
                if counted_memories is not None:
                    counted_memories.add(mem_name)
                return self.memory_models[mem_name].cacheArea_mm2 * 1e6  # mm² → um²
            else:
                function = "Register16"

        # Unified memory/fifo nodes from physical design netlist
        if function in ("memory", "fifo"):
            mem_name = node_data.get("name", "N/A") if node_data else "N/A"
            if mem_name != "N/A" and mem_name in self.memory_models:
                if counted_memories is not None and mem_name in counted_memories:
                    return 0.0
                if counted_memories is not None:
                    counted_memories.add(mem_name)
                return self.memory_models[mem_name].cacheArea_mm2 * 1e6  # mm² → um²
            return 0.0  # top_interface or unknown memories

        if function in ["N/A", "Call"]:
            return 0.0
        tv = self.tech_model.base_params.tech_values
        area_coeff = self.area_coeffs[function] * 500/7 # TODO: arbitrary, should fix the area coeffs at some point
        area = area_coeff * self.tech_model.base_params.area
        pipeline_cost = DATA_WIDTH * self.DFF_AREA * (self._latency(function)/self.tech_model.base_params.clk_period) # DATA_WIDTH DFFs needed for each extra cycle
        return area + pipeline_cost