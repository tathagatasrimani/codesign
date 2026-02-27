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


def _make_zero_bd():
    """Create a zero-initialized latency breakdown dict.

    The breakdown dict tracks how the delay to a given DFG node is split
    across four physical categories:
        clk    - register/pipeline-stage delays (one clk_period per resource
                 edge) and loop initiation-interval stalls (delay_1x *
                 (trip_count - 1)).  Represents time spent waiting on clock
                 boundaries rather than doing useful computation.
        logic  - combinational delay through arithmetic/logic units.
        memory - latency of cache-hit reads or cache-write operations to named
                 memory blocks (e.g. SRAM, DRAM caches).
        wire   - RC delay along physical wire segments in the netlist.
        memory_by_block - {mem_name: ns} sub-breakdown of the 'memory' total,
                 one entry per named memory instance.

    Percentages are computed relative to the total (sum of clk+logic+memory+wire).
    """
    return {"clk": 0.0, "logic": 0.0, "memory": 0.0, "wire": 0.0, "memory_by_block": {}}


def _add_bd(base, delta):
    """Return a new breakdown dict equal to base + delta.

    The four scalar fields are summed directly; memory_by_block dicts are
    merged with per-key summation so individual memory contributions accumulate
    correctly as we propagate breakdowns along the critical path.
    """
    result = {k: base[k] + delta[k] for k in ("clk", "logic", "memory", "wire")}
    mb = dict(base["memory_by_block"])
    for mem, val in delta["memory_by_block"].items():
        mb[mem] = mb.get(mem, 0.0) + val
    result["memory_by_block"] = mb
    return result


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
        logic_unit_models: Dict[str, Any] = None,
        ram_recurrences: Dict[str, Any] = None,
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
        self.logic_unit_models = logic_unit_models or {}
        self.ram_recurrences = ram_recurrences or {}
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
        # Latency breakdown (critical path ns and % by category)
        self.latency_breakdown = {"clk": 0.0, "logic": 0.0, "memory": 0.0, "wire": 0.0}
        self.latency_breakdown_pct = {"clk": 0.0, "logic": 0.0, "memory": 0.0, "wire": 0.0}
        self.latency_memory_by_block: Dict[str, float] = {}
        self.latency_memory_by_block_pct: Dict[str, float] = {}
        self.total_logic_ops = 0
        self.total_memory_ops = 0
        # Active energy breakdown (nJ and % by category)
        self.active_energy_breakdown = {"logic": 0.0, "memory": 0.0, "wire": 0.0}
        self.active_energy_breakdown_pct = {"logic": 0.0, "memory": 0.0, "wire": 0.0}
        self.active_energy_memory_by_block: Dict[str, float] = {}
        self.active_energy_memory_by_block_pct: Dict[str, float] = {}
        # Passive power breakdown (W and % by category; wires excluded)
        self.passive_power_breakdown = {"logic": 0.0, "memory": 0.0}
        self.passive_power_breakdown_pct = {"logic": 0.0, "memory": 0.0}
        self.passive_power_memory_by_block: Dict[str, float] = {}
        self.passive_power_memory_by_block_pct: Dict[str, float] = {}

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
            logic_unit_models=hw.logic_unit_models,
            ram_recurrences=hw.ram_recurrences,
        )

    def set_params_from_design_point(self, design_point: Dict[str, Any]):
        """Update tech, memory, and per-FU logic models from a design point.

        Expects design_point = {"logic": {...}, "memory": {...}, "fu_logic": {...}}.
        Falls back to treating the whole dict as logic params if "logic" key is absent.
        """
        self.tech_model.set_params_from_design_point(design_point)
        memory_config = design_point.get("memory", {})
        for mem_name, memory_model in self.memory_models.items():
            if mem_name in memory_config:
                memory_model.set_design_point(memory_config[mem_name])
        fu_logic_config = design_point.get("fu_logic", {})
        for rsc_name, lum in self.logic_unit_models.items():
            if rsc_name in fu_logic_config:
                lum.set_design_point(fu_logic_config[rsc_name])

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

        # ---------- Latency breakdown ----------
        # _calculate_execution_time_recursive propagates a breakdown dict
        # alongside every node's arrival time.  At the graph's end node the
        # accumulated dict represents the full critical-path delay split by
        # category.  Percentages are relative to the total critical-path length.
        if self.top_block_name not in self.dataflow_blocks:
            _graph_end = f"graph_end_{self.top_block_name}"
        else:
            _graph_end = f"{self.top_block_name}_graph_end_{self.top_block_name}"
        _bd = self.node_delay_breakdown[self.top_block_name]["full"].get(_graph_end, _make_zero_bd())
        self.latency_breakdown = {k: _bd[k] for k in ("clk", "logic", "memory", "wire")}
        self.latency_memory_by_block = dict(_bd["memory_by_block"])
        _total_lat = sum(self.latency_breakdown.values())
        if _total_lat > 0:
            self.latency_breakdown_pct = {k: v / _total_lat for k, v in self.latency_breakdown.items()}
            self.latency_memory_by_block_pct = {m: v / _total_lat for m, v in self.latency_memory_by_block.items()}

        # ---------- Op counts ----------
        # Counts all logic and memory ops executed at runtime.  Loop bodies are
        # multiplied by their full trip count (not count-1) because we want
        # actual execution count, not the pipelined-overlap adjustment used for
        # latency/energy.
        self.total_logic_ops, self.total_memory_ops = self._count_ops()

        # ---------- Active energy breakdown ----------
        # Accumulated in-place during calculate_active_energy; normalised here.
        # Percentages are relative to total active switching energy.
        _total_ae = sum(self.active_energy_breakdown.values())
        if _total_ae > 0:
            self.active_energy_breakdown_pct = {k: v / _total_ae for k, v in self.active_energy_breakdown.items()}
            self.active_energy_memory_by_block_pct = {m: v / _total_ae for m, v in self.active_energy_memory_by_block.items()}

        # ---------- Passive (leakage) power breakdown ----------
        # Accumulated in-place during calculate_passive_energy; wires carry no
        # leakage so only logic and memory units are tracked.
        _total_pp = sum(self.passive_power_breakdown.values())
        if _total_pp > 0:
            self.passive_power_breakdown_pct = {k: v / _total_pp for k, v in self.passive_power_breakdown.items()}
            self.passive_power_memory_by_block_pct = {m: v / _total_pp for m, v in self.passive_power_memory_by_block.items()}

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
        self.node_delay_breakdown = {}
        self.graph_delays = {}
        self.loop_delays_1x = {}  # loop_name -> delay of one iteration (ns)
        self.loop_ii_info = {}  # loop_name -> {II, resource_II, recurrence_II, bottleneck, critical_recurrence_node, critical_recurrence_ns}

        for basic_block_name in self.scheduled_dfgs:
            self.node_arrivals[basic_block_name] = {"full": {}, "loop_1x": {}, "loop_2x": {}, "loop_1x_full": {}}
            self.node_delay_breakdown[basic_block_name] = {"full": {}, "loop_1x": {}, "loop_2x": {}, "loop_1x_full": {}}

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

        Algorithm — longest-path with parallel breakdown tracking
        ---------------------------------------------------------
        The DFG is a DAG where each node represents an operation and each edge
        carries a delay.  We perform a single forward pass over the nodes (they
        are stored in topological order) and track, for every node, both:

            node_arrivals[block][type][node]        — scalar arrival time (ns)
            node_delay_breakdown[block][type][node] — breakdown dict (see
                                                      _make_zero_bd)

        For each predecessor edge we compute a (pred_delay, pred_bd_delta) pair
        and update the node's arrival if the new candidate is strictly greater.
        When a new maximum is found, the node's breakdown is replaced with
        pred_breakdown + pred_bd_delta, so the breakdown always reflects the
        actual critical path that determined the arrival time.

        Edge-type to breakdown-category mapping
        ----------------------------------------
        resource_edge AND pred.function == "II"  →  clk
            The II node represents a loop that runs `count` iterations.  The
            outer graph sees a delay of  delay_1x * (count - 1)  where delay_1x
            is the critical-path latency of a single loop iteration (computed
            by a recursive call with resource_delays_only=True).  This whole
            quantity is classified as 'clk' because it represents initiation-
            interval stalls between successive resource uses — time governed by
            the clock, not by logic depth.  Op counts are separately multiplied
            by the full trip count `count` in _count_ops_in_dfg.

        resource_edge AND pred.function != "II"  →  clk
            A plain register/pipeline-stage edge costs one clock period.

        pred.function == "Call"  →  propagate sub-function's breakdown
            The delay equals the called function's total latency.  Rather than
            re-categorising it, we copy the end-node breakdown of the called
            function's graph so that its clk/logic/memory/wire split flows up
            into the caller's critical path.

        pred.function == "Wire"  →  wire
            RC wire delay computed from segment lengths and layer parasitics.

        pred.function in _MEMORY_OPS (with named memory)  →  memory
            Cache-hit or write latency from the memory model.  Also recorded
            in memory_by_block[mem_name] for per-instance breakdown.

        everything else  →  logic
            Combinational delay via logical-effort coefficients (gamma table).
            Register ops (mem_name == 'N/A') fall through to Register16 and
            are also counted here.

        Args:
            basic_block_name: Name of the basic block
            dfg: NetworkX DiGraph representing the DFG
            graph_end_node: Name of the end node
            graph_type: Type of graph ('full', 'loop_1x', 'loop_2x')
            resource_delays_only: If True, only count resource edge delays
                (used when computing delay_1x for II nodes)

        Returns:
            Execution time in ns (arrival time at graph_end_node)
        """
        log_info(f"calculating execution time for {basic_block_name} with graph end node {graph_end_node}")

        tv = self.tech_model.base_params.tech_values

        # Initialize all node arrivals and delay breakdowns to 0
        for node in dfg.nodes:
            self.node_arrivals[basic_block_name][graph_type][node] = 0.0
            self.node_delay_breakdown[basic_block_name][graph_type][node] = _make_zero_bd()

        # Process nodes
        for node in dfg.nodes:
            preds = list(dfg.predecessors(node))

            for pred in preds:
                pred_delay = 0.0
                pred_bd_delta = _make_zero_bd()

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
                        clk_period = sim_util.xreplace_safe(self.tech_model.base_params.clk_period, tv)
                        resource_II = math.ceil(delay_1x / clk_period) if clk_period > 0 else 1

                        ram_recurrences = self.ram_recurrences.get(basic_block_name, {}).get(loop_name, {})
                        log_info(f"ram_recurrences: {ram_recurrences} for loop {loop_name}")
                        recurrence_II = 0
                        crit_path = None
                        for mem_name, recurrence_info in ram_recurrences.items():
                            path_latency = 0
                            for node in recurrence_info["path"]:
                                latency = self._latency(dfg.nodes[node]["function"], dfg.nodes[node])
                                log_info(f"latency: {latency} for node {node}")
                                path_latency += math.ceil(latency / clk_period)
                            store_node = recurrence_info["path"][-1]
                            ram_depth = recurrence_info["depth"]
                            assert ram_depth > 0
                            candidate_II = math.ceil(path_latency / (ram_depth))
                            if candidate_II > recurrence_II:
                                recurrence_II = candidate_II
                                crit_path = recurrence_info["path"]
                            log_info(f"recurrence_II: {candidate_II} for store node {store_node} (mem {mem_name}) with ram depth {ram_depth}")

                        II_cycles = max(resource_II, recurrence_II)
                        pred_delay = II_cycles * clk_period * (int(dfg.nodes[pred]["count"]) - 1)
                        pred_bd_delta["clk"] = pred_delay
                        self.loop_ii_info[loop_name] = {
                            "II": II_cycles,
                            "resource_II": resource_II,
                            "recurrence_II": recurrence_II,
                            "delay_1x_ns": delay_1x,
                            "bottleneck": "resource" if resource_II >= recurrence_II else "recurrence",
                            "critical_recurrence_path": crit_path,
                        }
                    else:
                        pred_delay = sim_util.xreplace_safe(
                            self.tech_model.base_params.clk_period, tv
                        )
                        pred_bd_delta["clk"] = pred_delay

                elif dfg.nodes[pred]["function"] == "Call":
                    if basic_block_name in self.dataflow_blocks:
                        pred_delay = 0.0 # ignore calls because operations are inlined
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
                        pred_bd_delta = dict(
                            self.node_delay_breakdown[call_fn]["full"][f"graph_end_{call_fn}"]
                        )
                        pred_bd_delta["memory_by_block"] = dict(pred_bd_delta["memory_by_block"])

                elif not resource_delays_only:
                    if dfg.nodes[pred]["function"] == "Wire":
                        src = dfg.nodes[pred]["src_node"]
                        dst = dfg.nodes[pred]["dst_node"]
                        rsc_edge = self._get_rsc_edge((src, dst), dfg)
                        if rsc_edge in self.edge_to_nets:
                            pred_delay = self._wire_delay(rsc_edge)
                            pred_bd_delta["wire"] = pred_delay
                            log_info(f"added wire delay {pred_delay} for edge {rsc_edge}")
                        else:
                            log_info(f"no wire delay for edge {rsc_edge}")
                    else:
                        pred_delay = self._latency(dfg.nodes[pred]["function"], dfg.nodes[pred])
                        fn = dfg.nodes[pred]["function"]
                        mem_name = dfg.nodes[pred].get("mem_name", "N/A")
                        if fn in _MEMORY_OPS and mem_name != "N/A" and mem_name in self.memory_models:
                            pred_bd_delta["memory"] = pred_delay
                            pred_bd_delta["memory_by_block"][mem_name] = pred_delay
                        else:
                            pred_bd_delta["logic"] = pred_delay

                log_info(f"pred_delay: {pred_delay} for node {node} and pred {pred}")

                # Update node arrival time and breakdown (max of all predecessor paths)
                arrival = self.node_arrivals[basic_block_name][graph_type][pred] + pred_delay
                current = self.node_arrivals[basic_block_name][graph_type][node]
                if arrival > current:
                    self.node_arrivals[basic_block_name][graph_type][node] = arrival
                    pred_bd = self.node_delay_breakdown[basic_block_name][graph_type][pred]
                    self.node_delay_breakdown[basic_block_name][graph_type][node] = _add_bd(pred_bd, pred_bd_delta)

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

    def _get_fu_params(self, fn: str, node_data):
        """Return (delay, E_act_inv, P_pass_inv, area) for a node."""
        if "rsc" in node_data:
            rsc = node_data["rsc"] # if working with dfg
        else:
            rsc = node_data["name"] # if working with netlist
        assert rsc is not None, f"no per-FU model for fn={fn}, rsc={rsc}, node_data: {node_data}"
        #logger.info(f"logic_unit_models: {self.logic_unit_models}")
        if rsc not in self.logic_unit_models:
            if fn == "Register16" and node_data.get("mem_name") in self.logic_unit_models: # registers sometimes treated differently in netlist
                rsc = node_data.get("mem_name")
            else:
                log_info(f"no per-FU model for fn={fn}, rsc={rsc}, node_data: {node_data}") # TODO fix this issue
                return 0.0, 0.0, 0.0, 0.0
        lum = self.logic_unit_models[rsc]
        return lum.delay, lum.E_act_inv, lum.P_pass_inv, lum.area

    def _latency(self, op_type: str, node_data) -> float:
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
        delay, _, _, _ = self._get_fu_params(op_type, node_data)
        return math.ceil(self.gamma[op_type] * delay)

    def _count_ops(self) -> tuple:
        """Count total logic and memory operations across all scheduled DFGs.

        Loop bodies are multiplied by their full trip count so the result
        reflects the total number of operations executed at runtime, not the
        static op count in the IR.  Returns (total_logic_ops, total_memory_ops).
        """
        total_logic, total_memory = 0, 0
        for block_name in self.scheduled_dfgs:
            logic, memory = self._count_ops_in_dfg(block_name, self.scheduled_dfgs[block_name])
            total_logic += logic
            total_memory += memory
        return total_logic, total_memory

    def _count_ops_in_dfg(self, block_name: str, dfg) -> tuple:
        """Recursively count ops in dfg, scaling loop bodies by trip count."""
        logic, memory = 0, 0
        for node, data in dfg.nodes(data=True):
            fn = data["function"]
            if fn == "II":
                count = int(data["count"])
                loop_name = data["loop_name"]
                l, m = self._count_ops_in_dfg(block_name, self.loop_1x_graphs[loop_name][False])
                logic += l * count
                memory += m * count
            elif fn in _MEMORY_OPS:
                mem_name = data.get("mem_name", "N/A")
                if mem_name != "N/A" and mem_name in self.memory_models:
                    memory += 1
            elif fn not in ("Wire", "Call", "N/A") and fn in self.gamma:
                logic += 1
        return logic, memory

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
        self.active_energy_breakdown = {"logic": 0.0, "memory": 0.0, "wire": 0.0}
        self.active_energy_memory_by_block = {}
        total_active_energy = 0.0
        for basic_block_name in self.scheduled_dfgs:
            total_active_energy += self._calculate_active_energy_basic_block(
                basic_block_name,
                self.scheduled_dfgs[basic_block_name],
                breakdown=self.active_energy_breakdown,
                memory_by_block=self.active_energy_memory_by_block,
            )
        return total_active_energy

    def _calculate_active_energy_basic_block(self, basic_block_name: str, dfg,
                                              breakdown=None, memory_by_block=None) -> float:
        """
        Calculate active energy for a single basic block.

        Replicates HardwareModel.calculate_active_energy_basic_block.
        breakdown and memory_by_block, if provided, are accumulated in-place.
        """
        tv = self.tech_model.base_params.tech_values
        total_active_energy_basic_block = 0.0
        loop_count = 1
        loop_energy = 0.0

        for node, data in dfg.nodes(data=True):
            if data["function"] == "II":
                loop_count = int(data["count"])
                loop_name = data["loop_name"]
                loop_bd = {"logic": 0.0, "memory": 0.0, "wire": 0.0} if breakdown is not None else None
                loop_mbb = {} if memory_by_block is not None else None
                loop_energy = self._calculate_active_energy_basic_block(
                    basic_block_name,
                    self.loop_1x_graphs[loop_name][False],
                    breakdown=loop_bd,
                    memory_by_block=loop_mbb,
                )
                total_active_energy_basic_block += loop_energy * (loop_count - 1)
                if breakdown is not None:
                    for k in breakdown:
                        breakdown[k] += loop_bd[k] * (loop_count - 1)
                if memory_by_block is not None:
                    for m, v in loop_mbb.items():
                        memory_by_block[m] = memory_by_block.get(m, 0.0) + v * (loop_count - 1)
                log_info(f"loop count for {basic_block_name}: {loop_count}")
                log_info(f"loop energy for {basic_block_name}: {sim_util.xreplace_safe(loop_energy, tv)}")

            elif data["function"] == "Wire":
                src = data["src_node"]
                dst = data["dst_node"]
                rsc_edge = self._get_rsc_edge((src, dst), dfg)
                if rsc_edge in self.edge_to_nets:
                    energy = self._wire_energy(rsc_edge)
                    total_active_energy_basic_block += energy
                    if breakdown is not None:
                        breakdown["wire"] += energy
                    log_info(f"edge {rsc_edge} is in edge_to_nets")
                else:
                    log_info(f"edge {rsc_edge} is not in edge_to_nets")

            else:
                energy = self._symbolic_energy_active(data["function"], data)
                total_active_energy_basic_block += energy
                if breakdown is not None:
                    fn = data["function"]
                    mem_name = data.get("mem_name", "N/A")
                    if fn in _MEMORY_OPS and mem_name != "N/A" and mem_name in self.memory_models:
                        breakdown["memory"] += energy
                        if memory_by_block is not None:
                            memory_by_block[mem_name] = memory_by_block.get(mem_name, 0.0) + energy
                    else:
                        breakdown["logic"] += energy
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

        delay, E_act_inv, _, _ = self._get_fu_params(function, node_data)
        alpha = self.alpha[function]

        # Unpipelined energy
        unpipelined_energy = alpha * E_act_inv

        # Pipeline cost (DFF energy for additional cycles); DFF_ENERGY uses per-FU E_act_inv
        DFF_ENERGY = 20 * E_act_inv
        clk_period = sim_util.xreplace_safe(self.tech_model.base_params.clk_period,
                                             self.tech_model.base_params.tech_values)

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
        self.passive_power_breakdown = {"logic": 0.0, "memory": 0.0}
        self.passive_power_memory_by_block = {}

        for node, data in self.netlist.nodes(data=True):
            power = self._symbolic_power_passive(data["function"], data, counted_memories)
            total_passive_power += power
            # Categorize for breakdown
            fn = data["function"]
            mem_name = data.get("mem_name", "N/A") if data else "N/A"
            if fn in _MEMORY_OPS:
                if mem_name != "N/A" and mem_name in self.memory_models:
                    self.passive_power_breakdown["memory"] += power
                    if power > 0:
                        self.passive_power_memory_by_block[mem_name] = (
                            self.passive_power_memory_by_block.get(mem_name, 0.0) + power
                        )
                else:
                    self.passive_power_breakdown["logic"] += power  # Register16
            elif fn in ("memory", "fifo"):
                self.passive_power_breakdown["memory"] += power
                blk = data.get("name", "N/A") if data else "N/A"
                if power > 0 and blk != "N/A":
                    self.passive_power_memory_by_block[blk] = (
                        self.passive_power_memory_by_block.get(blk, 0.0) + power
                    )
            elif fn not in ("N/A", "Call") and fn in self.beta:
                self.passive_power_breakdown["logic"] += power
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

        delay, _, P_pass_inv, _ = self._get_fu_params(function, node_data)
        beta = self.beta[function]

        # Unpipelined passive power
        unpipelined_power = beta * P_pass_inv

        # Pipeline cost (DFF passive power for additional cycles); uses per-FU P_pass_inv
        DFF_PASSIVE_POWER = 20 * P_pass_inv
        clk_period = sim_util.xreplace_safe(self.tech_model.base_params.clk_period,
                                             self.tech_model.base_params.tech_values)

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
        _, _, _, fu_area = self._get_fu_params(function, node_data)
        area_coeff = self.area_coeffs[function] * 500/7 # TODO: arbitrary, should fix the area coeffs at some point
        area = area_coeff * fu_area
        pipeline_cost = DATA_WIDTH * self.DFF_AREA * (self._latency(function, node_data)/self.tech_model.base_params.clk_period) # DATA_WIDTH DFFs needed for each extra cycle
        return area + pipeline_cost