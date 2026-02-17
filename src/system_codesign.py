"""
System-level codesign orchestrator.

Runs per-block codesign for each unique block type, then composes
results at the system level using DFG scheduling to find the
globally optimal system design.

Usage:
    python3 -m src.system_codesign --system-config transformer_1layer_system
"""

import argparse
import datetime
import itertools
import json
import logging
import math
import os
import pickle
import subprocess
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import networkx as nx

from src.inverse_pass.utils import DesignPointResult

logger = logging.getLogger("system_codesign")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BlockCodesignResult:
    """Results from running codesign on a single block type."""
    block_name: str
    config_name: str
    save_dir: str
    all_results: List[DesignPointResult]
    valid_results: List[DesignPointResult]
    best_result: Optional[DesignPointResult]


@dataclass
class GlobalMemoryModel:
    """Off-chip global memory (DRAM) model for data movement costs."""
    bandwidth_gbps: float = 100.0       # GB/s
    latency_ns: float = 100.0           # per-access latency (ns)
    energy_per_bit_pj: float = 10.0     # pJ per bit transferred
    capacity_bytes: float = 16e9        # total DRAM capacity

    def transfer_time_ns(self, size_bytes: float) -> float:
        """Time to transfer size_bytes through global memory."""
        if size_bytes <= 0:
            return 0.0
        bw_bytes_per_ns = self.bandwidth_gbps  # GB/s = bytes/ns
        return (size_bytes / bw_bytes_per_ns) + self.latency_ns

    def transfer_energy_nj(self, size_bytes: float) -> float:
        """Energy to transfer size_bytes through global memory."""
        if size_bytes <= 0:
            return 0.0
        return size_bytes * 8 * self.energy_per_bit_pj * 1e-3  # pJ -> nJ


@dataclass
class BlockDataSizes:
    """Data size information for a block type."""
    input_bytes: float = 0.0
    output_bytes: float = 0.0
    weight_bytes: float = 0.0
    kv_cache_bytes: float = 0.0         # per-layer KV cache (attention only)
    shared_weights: bool = False         # if True, weights loaded once for all repetitions


@dataclass
class EdgeTransferConfig:
    """Configuration for data transfer along a DFG edge."""
    transfer: str = "global_memory"      # "global_memory" | "on_chip"
    on_chip_latency_ns: float = 10.0
    on_chip_energy_per_bit_pj: float = 0.5


@dataclass
class OverheadBreakdown:
    """Breakdown of inter-block overhead costs."""
    weight_load_time_ns: float = 0.0
    weight_load_energy_nj: float = 0.0
    comm_time_ns: float = 0.0
    comm_energy_nj: float = 0.0
    kv_cache_time_ns: float = 0.0
    kv_cache_energy_nj: float = 0.0

    @property
    def total_time_ns(self) -> float:
        return self.weight_load_time_ns + self.comm_time_ns + self.kv_cache_time_ns

    @property
    def total_energy_nj(self) -> float:
        return self.weight_load_energy_nj + self.comm_energy_nj + self.kv_cache_energy_nj


@dataclass
class SystemDesignPoint:
    """A complete system-level design: one DesignPointResult per block type."""
    block_assignments: Dict[str, DesignPointResult]
    system_execution_time: float = 0.0    # ns (compute + overhead)
    system_total_energy: float = 0.0      # nJ (compute + overhead)
    system_total_area: float = 0.0        # um^2
    system_obj_value: float = 0.0
    satisfies_system_constraints: bool = False
    # Overhead breakdown
    compute_time_ns: float = 0.0
    compute_energy_nj: float = 0.0
    overhead: Optional[OverheadBreakdown] = None


# ---------------------------------------------------------------------------
# System Codesign orchestrator
# ---------------------------------------------------------------------------

class SystemCodesign:
    def __init__(self, system_config_path: str, system_config_name: str,
                 block_results_dir: Optional[str] = None):
        self.codesign_root_dir = os.getcwd()
        self.system_cfg = self._load_system_config(system_config_path, system_config_name)
        self.block_results: Dict[str, BlockCodesignResult] = {}
        self.block_results_dir = block_results_dir

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(
            "logs", "system", f"{timestamp}_{self.system_cfg['name']}"
        )
        os.makedirs(self.save_dir, exist_ok=True)

        # Set up logging to file
        fh = logging.FileHandler(os.path.join(self.save_dir, "system_codesign.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)

        logger.info(f"System config: {self.system_cfg['name']}")
        logger.info(f"Save directory: {self.save_dir}")
        if block_results_dir:
            logger.info(f"Loading block results from previous run: {block_results_dir}")

        # Parse global memory model
        mem_cfg = self.system_cfg.get("global_memory", {})
        self.global_memory = GlobalMemoryModel(
            bandwidth_gbps=float(mem_cfg.get("bandwidth_gbps", 100.0)),
            latency_ns=float(mem_cfg.get("latency_ns", 100.0)),
            energy_per_bit_pj=float(mem_cfg.get("energy_per_bit_pj", 10.0)),
            capacity_bytes=float(mem_cfg.get("capacity_bytes", 16e9)),
        )
        logger.info(
            f"Global memory: BW={self.global_memory.bandwidth_gbps} GB/s, "
            f"latency={self.global_memory.latency_ns} ns, "
            f"energy={self.global_memory.energy_per_bit_pj} pJ/bit"
        )

        # Parse per-block data sizes (explicit or inferred)
        self.block_data_sizes: Dict[str, BlockDataSizes] = {}
        self._init_block_data_sizes()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_system_config(config_path: str, config_name: str) -> dict:
        import yaml
        with open(config_path, "r") as f:
            all_configs = yaml.safe_load(f)
        if config_name not in all_configs:
            available = list(all_configs.keys())
            raise ValueError(
                f"System config '{config_name}' not found in {config_path}. "
                f"Available: {available}"
            )
        return all_configs[config_name]

    # ------------------------------------------------------------------
    # Block data size initialization
    # ------------------------------------------------------------------

    def _init_block_data_sizes(self):
        """Initialize data sizes for each block type (explicit YAML or inferred)."""
        for block_name, block_cfg in self.system_cfg["block_types"].items():
            if "data_sizes" in block_cfg:
                ds = block_cfg["data_sizes"]
                self.block_data_sizes[block_name] = BlockDataSizes(
                    input_bytes=float(ds.get("input_bytes", 0)),
                    output_bytes=float(ds.get("output_bytes", 0)),
                    weight_bytes=float(ds.get("weight_bytes", 0)),
                    kv_cache_bytes=float(ds.get("kv_cache_bytes", 0)),
                    shared_weights=block_cfg.get("shared_weights", False),
                )
                logger.info(
                    f"Block '{block_name}' data sizes (from YAML): "
                    f"in={self.block_data_sizes[block_name].input_bytes:.0f}B, "
                    f"out={self.block_data_sizes[block_name].output_bytes:.0f}B, "
                    f"weights={self.block_data_sizes[block_name].weight_bytes:.0f}B, "
                    f"kv_cache={self.block_data_sizes[block_name].kv_cache_bytes:.0f}B"
                )
            else:
                # Try to infer from data.py
                inferred = self._infer_block_data_sizes(block_name, block_cfg)
                self.block_data_sizes[block_name] = inferred
                logger.info(
                    f"Block '{block_name}' data sizes (inferred): "
                    f"in={inferred.input_bytes:.0f}B, "
                    f"out={inferred.output_bytes:.0f}B, "
                    f"weights={inferred.weight_bytes:.0f}B, "
                    f"kv_cache={inferred.kv_cache_bytes:.0f}B"
                )

    def _infer_block_data_sizes(self, block_name: str, block_cfg: dict) -> BlockDataSizes:
        """Infer data sizes by loading the benchmark model from data.py."""
        try:
            benchmark_name = self._resolve_benchmark_name(block_cfg["codesign_config"])
            if not benchmark_name:
                logger.warning(f"Could not resolve benchmark name for '{block_name}', using zero sizes")
                return BlockDataSizes(shared_weights=block_cfg.get("shared_weights", False))

            import torch
            import sys

            # Import data.py to get model config and input shapes
            data_py_path = os.path.join(self.codesign_root_dir, "Stream-HLS", "examples")
            if data_py_path not in sys.path:
                sys.path.insert(0, data_py_path)

            from data import model_configs
            # Search across all benchmark groups for this benchmark name
            model_entry = None
            for group_name, group in model_configs.items():
                if benchmark_name in group:
                    model_entry = group[benchmark_name]
                    break

            if model_entry is None:
                logger.warning(f"Benchmark '{benchmark_name}' not found in data.py, using zero sizes")
                return BlockDataSizes(shared_weights=block_cfg.get("shared_weights", False))

            # Import and instantiate the model
            pymodels_path = os.path.join(data_py_path, "pymodels", "transformers")
            if pymodels_path not in sys.path:
                sys.path.insert(0, pymodels_path)

            model_class_name = model_entry["class"]
            model_module = __import__(model_class_name)
            model_class = getattr(model_module, model_class_name)
            model = model_class(**model_entry.get("config", {}))
            model.eval()

            # Compute weight bytes
            weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

            # Compute input bytes
            input_tensors = model_entry["input"]
            input_bytes = sum(t.numel() * t.element_size() for t in input_tensors)

            # Compute output bytes via forward pass
            with torch.no_grad():
                output = model(*input_tensors)
            if isinstance(output, torch.Tensor):
                output_bytes = output.numel() * output.element_size()
            elif isinstance(output, (tuple, list)):
                output_bytes = sum(o.numel() * o.element_size() for o in output if isinstance(o, torch.Tensor))
            else:
                output_bytes = input_bytes  # fallback: assume same as input

            # Estimate KV cache for attention blocks
            kv_cache_bytes = 0.0
            config = model_entry.get("config", {})
            if "n_heads" in config or "num_heads" in config:
                # This is an attention block — estimate KV cache
                n_heads = config.get("n_heads", config.get("num_heads", 1))
                dim = config.get("dim", config.get("embed_dim", 256))
                head_dim = dim // n_heads
                seq_len = input_tensors[0].shape[1] if len(input_tensors[0].shape) >= 2 else 1
                batch_size = input_tensors[0].shape[0]
                elem_size = input_tensors[0].element_size()
                # KV cache: 2 (K + V) * batch * seq_len * n_heads * head_dim * elem_size
                kv_cache_bytes = 2 * batch_size * seq_len * n_heads * head_dim * elem_size

            return BlockDataSizes(
                input_bytes=float(input_bytes),
                output_bytes=float(output_bytes),
                weight_bytes=float(weight_bytes),
                kv_cache_bytes=float(kv_cache_bytes),
                shared_weights=block_cfg.get("shared_weights", False),
            )

        except Exception as e:
            logger.warning(f"Failed to infer data sizes for '{block_name}': {e}")
            return BlockDataSizes(shared_weights=block_cfg.get("shared_weights", False))

    def _resolve_benchmark_name(self, config_name: str) -> Optional[str]:
        """Resolve the benchmark name from a codesign config by walking the base_cfg chain."""
        import yaml

        # Load all config sources
        all_cfgs = {}
        cfg_files = [
            os.path.join(self.codesign_root_dir, "src", "yaml", "codesign_cfg.yaml"),
            os.path.join(self.codesign_root_dir, "test", "additional_configs", "patrick_sweep_configs.yaml"),
        ]
        for cfg_file in cfg_files:
            if os.path.exists(cfg_file):
                with open(cfg_file, "r") as f:
                    loaded = yaml.safe_load(f) or {}
                    all_cfgs.update(loaded)

        # Walk the config chain to find 'benchmark' in 'args'
        current = config_name
        visited = set()
        while current and current not in visited:
            visited.add(current)
            cfg = all_cfgs.get(current, {})
            args = cfg.get("args", {})
            if "benchmark" in args:
                return args["benchmark"]
            current = cfg.get("base_cfg")

        return None

    # ------------------------------------------------------------------
    # Per-block codesign execution
    # ------------------------------------------------------------------

    def _build_block_cmd(self, block_name: str, block_cfg: dict, block_save_dir: str) -> list:
        """Build the CLI command for a single block codesign run."""
        config_name = block_cfg["codesign_config"]
        overrides = block_cfg.get("config_overrides", {})

        cmd = [
            "python3", "-m", "src.codesign",
            "--config", config_name,
            "--savedir", block_save_dir,
        ]

        # Pass through checkpoint args from system config if present
        if "checkpoint_load_dir" in block_cfg:
            cmd.extend(["--checkpoint_load_dir", str(block_cfg["checkpoint_load_dir"])])
        if "checkpoint_start_step" in block_cfg:
            cmd.extend(["--checkpoint_start_step", str(block_cfg["checkpoint_start_step"])])
        if "stop_at_checkpoint" in block_cfg:
            cmd.extend(["--stop_at_checkpoint", str(block_cfg["stop_at_checkpoint"])])

        for key, value in overrides.items():
            cmd.extend([f"--{key}", str(value)])

        return cmd

    def run_per_block_codesign(self):
        """Run codesign for each unique block type via subprocess.

        If --block-results-dir was provided, skip running codesign and load
        results from the previous system run directory instead.

        Block codesigns are launched in parallel as independent subprocesses.
        """
        if self.block_results_dir:
            logger.info("Skipping per-block codesign — loading from previous run")
            for block_name in self.system_cfg["block_types"]:
                block_save_dir = os.path.join(self.block_results_dir, f"block_{block_name}")
                if not os.path.isdir(block_save_dir):
                    raise FileNotFoundError(
                        f"Expected block results at {block_save_dir} but directory not found"
                    )
                self._load_block_results(block_name, block_save_dir)
            return

        # Launch all block codesigns in parallel
        running: Dict[str, dict] = {}  # block_name -> {proc, save_dir, stdout_f, stderr_f}

        for block_name, block_cfg in self.system_cfg["block_types"].items():
            block_save_dir = os.path.join(self.save_dir, f"block_{block_name}")
            os.makedirs(block_save_dir, exist_ok=True)

            cmd = self._build_block_cmd(block_name, block_cfg, block_save_dir)
            logger.info(f"Launching codesign for block '{block_name}': {' '.join(cmd)}")

            stdout_path = os.path.join(block_save_dir, "codesign_stdout.log")
            stderr_path = os.path.join(block_save_dir, "codesign_stderr.log")
            stdout_f = open(stdout_path, "w")
            stderr_f = open(stderr_path, "w")

            proc = subprocess.Popen(
                cmd, stdout=stdout_f, stderr=stderr_f,
                cwd=self.codesign_root_dir,
            )
            running[block_name] = {
                "proc": proc,
                "save_dir": block_save_dir,
                "stdout_f": stdout_f,
                "stderr_f": stderr_f,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
            }

        logger.info(f"All {len(running)} block codesigns launched in parallel")

        # Wait for all to complete
        failures = []
        for block_name, info in running.items():
            proc = info["proc"]
            logger.info(f"Waiting for block '{block_name}' (pid {proc.pid})...")
            proc.wait()
            info["stdout_f"].close()
            info["stderr_f"].close()

            if proc.returncode != 0:
                # Read last 500 chars of logs for error reporting
                with open(info["stderr_path"], "r") as f:
                    stderr_tail = f.read()[-500:]
                with open(info["stdout_path"], "r") as f:
                    stdout_tail = f.read()[-500:]
                logger.error(f"Block '{block_name}' failed (exit code {proc.returncode})")
                logger.error(f"Last 500 chars of stderr: {stderr_tail}")
                logger.error(f"Last 500 chars of stdout: {stdout_tail}")
                failures.append(block_name)
            else:
                logger.info(f"Block '{block_name}' completed successfully")
                self._load_block_results(block_name, info["save_dir"])

        if failures:
            raise RuntimeError(
                f"Codesign failed for block(s): {', '.join(failures)}. "
                f"Check logs in {self.save_dir} for details."
            )

    @staticmethod
    def _to_float(value):
        """Convert a potentially symbolic value to float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            # SymPy expression that wasn't fully resolved — try .evalf()
            try:
                return float(value.evalf())
            except Exception:
                logger.warning(f"Could not convert {value} to float, using 0.0")
                return 0.0

    def _load_block_results(self, block_name: str, block_save_dir: str):
        """Load pickled DesignPointResult list from a completed block run."""
        # The save_dir from codesign.py is a timestamped subdirectory of block_save_dir
        # Find the most recent results pickle
        pattern = os.path.join(block_save_dir, "**", "all_design_point_results_*.pkl")
        results_files = sorted(glob.glob(pattern, recursive=True))

        if not results_files:
            raise FileNotFoundError(
                f"No results files found for block '{block_name}' in {block_save_dir}"
            )

        all_results = []
        for results_path in results_files:
            logger.info(f"Loading results from {results_path}")
            with open(results_path, "rb") as f:
                all_results.extend(pickle.load(f))

        # Sanitize: ensure numeric fields are plain floats (not SymPy expressions)
        for r in all_results:
            r.total_area = self._to_float(r.total_area)
            r.execution_time = self._to_float(r.execution_time)
            r.total_active_energy = self._to_float(r.total_active_energy)
            r.total_passive_energy = self._to_float(r.total_passive_energy)

        valid_results = [r for r in all_results if r.satisfies_constraints]
        sorted_valid = sorted(valid_results, key=lambda r: r.obj_value)
        best_result = sorted_valid[0] if sorted_valid else None

        self.block_results[block_name] = BlockCodesignResult(
            block_name=block_name,
            config_name=self.system_cfg["block_types"][block_name]["codesign_config"],
            save_dir=block_save_dir,
            all_results=all_results,
            valid_results=sorted_valid,
            best_result=best_result,
        )
        logger.info(
            f"Loaded {len(all_results)} results for block '{block_name}' "
            f"({len(valid_results)} valid)"
        )

    # ------------------------------------------------------------------
    # System DFG construction
    # ------------------------------------------------------------------

    def _get_edge_transfer_config(self, src_block_type: str, dst_block_type: str) -> EdgeTransferConfig:
        """Get the transfer config for an edge between two block types."""
        dfg_cfg = self.system_cfg.get("dfg", {})
        default_transfer = dfg_cfg.get("default_transfer", "global_memory")

        # Check for edge-specific overrides
        for override in dfg_cfg.get("edge_overrides", []):
            if override.get("src") == src_block_type and override.get("dst") == dst_block_type:
                return EdgeTransferConfig(
                    transfer=override.get("transfer", default_transfer),
                    on_chip_latency_ns=float(override.get("on_chip_latency_ns", 10.0)),
                    on_chip_energy_per_bit_pj=float(override.get("on_chip_energy_per_bit_pj", 0.5)),
                )

        return EdgeTransferConfig(transfer=default_transfer)

    def build_system_dfg(self) -> nx.DiGraph:
        """Build the system-level DFG from config with edge transfer configs."""
        G = nx.DiGraph()
        block_types = self.system_cfg["block_types"]
        dfg_cfg = self.system_cfg["dfg"]

        # Add nodes for all block instances
        for block_name, block_cfg in block_types.items():
            repeat = block_cfg.get("repeat_count", 1)
            for i in range(repeat):
                node_id = f"{block_name}[{i}]" if repeat > 1 else block_name
                G.add_node(node_id, block_type=block_name, instance_index=i)

        if dfg_cfg.get("pattern") == "chain":
            # Auto-generate chain: layer_blocks repeated
            layer_blocks = dfg_cfg["layer_blocks"]
            max_repeat = max(
                block_types[b].get("repeat_count", 1) for b in layer_blocks
            )
            prev_node = None
            prev_block_type = None
            for layer_idx in range(max_repeat):
                for block_name in layer_blocks:
                    repeat = block_types[block_name].get("repeat_count", 1)
                    if layer_idx < repeat:
                        node_id = (
                            f"{block_name}[{layer_idx}]"
                            if repeat > 1
                            else block_name
                        )
                        if prev_node is not None:
                            edge_cfg = self._get_edge_transfer_config(
                                prev_block_type, block_name
                            )
                            G.add_edge(prev_node, node_id, transfer_config=edge_cfg)
                        prev_node = node_id
                        prev_block_type = block_name
        else:
            # Explicit edges
            for src, dst in dfg_cfg.get("edges", []):
                # Resolve block types from node ids
                src_type = G.nodes[src]["block_type"] if src in G.nodes else src
                dst_type = G.nodes[dst]["block_type"] if dst in G.nodes else dst
                edge_cfg = self._get_edge_transfer_config(src_type, dst_type)
                G.add_edge(src, dst, transfer_config=edge_cfg)

        logger.info(f"System DFG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # ------------------------------------------------------------------
    # System-level optimization
    # ------------------------------------------------------------------

    def optimize_system(self, max_per_block: int = 50) -> Optional[SystemDesignPoint]:
        """Cross-product search over per-block pareto subsets."""
        block_types = self.system_cfg["block_types"]
        constraints = self.system_cfg.get("system_constraints", {})
        objective = constraints.get("objective", "edp")
        shared_tech = self.system_cfg.get("shared_tech", False)
        execution_model = self.system_cfg.get("execution_model", "sequential")

        # Select top candidates per block (sorted by obj_value)
        block_candidates: Dict[str, List[DesignPointResult]] = {}
        for block_name in block_types:
            br = self.block_results[block_name]
            block_candidates[block_name] = br.valid_results[:max_per_block]
            if not block_candidates[block_name]:
                logger.warning(f"No valid results for block '{block_name}'")

        block_names = sorted(block_candidates.keys())
        candidate_lists = [block_candidates[bn] for bn in block_names]

        # Skip if any block has no candidates
        if any(len(cl) == 0 for cl in candidate_lists):
            logger.warning("At least one block has no valid candidates; cannot optimize")
            return None

        total_combos = 1
        for cl in candidate_lists:
            total_combos *= len(cl)
        logger.info(
            f"System optimization: {' x '.join(str(len(cl)) for cl in candidate_lists)} "
            f"= {total_combos} combinations"
        )

        best_system: Optional[SystemDesignPoint] = None
        best_obj = math.inf
        evaluated = 0

        for combo in itertools.product(*candidate_lists):
            assignments = {bn: result for bn, result in zip(block_names, combo)}

            # If shared_tech, all blocks must have the same logic design point
            if shared_tech:
                logic_points = [
                    r.design_point.get("logic", r.design_point) for r in combo
                ]
                if not all(lp == logic_points[0] for lp in logic_points):
                    continue

            system_point = self._compose_system_metrics(
                assignments, execution_model, objective
            )
            system_point.satisfies_system_constraints = (
                self._check_system_constraints(system_point, constraints)
            )

            if (
                system_point.satisfies_system_constraints
                and system_point.system_obj_value < best_obj
            ):
                best_obj = system_point.system_obj_value
                best_system = system_point

            evaluated += 1
            if evaluated % 10000 == 0:
                logger.info(f"Evaluated {evaluated}/{total_combos} system combinations...")

        logger.info(f"Evaluated {evaluated} system combinations")
        if best_system:
            logger.info(f"Best system objective: {best_obj:.4e}")
        else:
            logger.warning("No valid system design found!")

        return best_system

    # ------------------------------------------------------------------
    # Overhead computation
    # ------------------------------------------------------------------

    def _compute_edge_transfer(self, edge_cfg: EdgeTransferConfig, data_bytes: float):
        """Compute time and energy for an edge transfer."""
        if data_bytes <= 0:
            return 0.0, 0.0

        if edge_cfg.transfer == "on_chip":
            time_ns = edge_cfg.on_chip_latency_ns
            energy_nj = data_bytes * 8 * edge_cfg.on_chip_energy_per_bit_pj * 1e-3
        else:  # global_memory
            time_ns = self.global_memory.transfer_time_ns(data_bytes)
            energy_nj = self.global_memory.transfer_energy_nj(data_bytes)

        return time_ns, energy_nj

    def _compute_node_overhead(self, node_id: str, assignments: Dict[str, DesignPointResult]):
        """Compute overhead for a single DFG node (weight load + incoming comm + KV cache)."""
        G = self.system_dfg
        block_type = G.nodes[node_id]["block_type"]
        instance_idx = G.nodes[node_id]["instance_index"]
        data_sizes = self.block_data_sizes.get(block_type, BlockDataSizes())

        weight_time = 0.0
        weight_energy = 0.0
        comm_time = 0.0
        comm_energy = 0.0
        kv_time = 0.0
        kv_energy = 0.0

        # Weight loading from global memory
        if data_sizes.weight_bytes > 0:
            if not data_sizes.shared_weights or instance_idx == 0:
                weight_time = self.global_memory.transfer_time_ns(data_sizes.weight_bytes)
                weight_energy = self.global_memory.transfer_energy_nj(data_sizes.weight_bytes)

        # Incoming activation transfers from predecessors
        for pred in G.predecessors(node_id):
            pred_block_type = G.nodes[pred]["block_type"]
            pred_data_sizes = self.block_data_sizes.get(pred_block_type, BlockDataSizes())
            edge_data = G.edges[pred, node_id]
            edge_cfg = edge_data.get("transfer_config", EdgeTransferConfig())

            # Transfer size = predecessor's output bytes
            transfer_bytes = pred_data_sizes.output_bytes
            t, e = self._compute_edge_transfer(edge_cfg, transfer_bytes)
            comm_time += t
            comm_energy += e

        # KV cache write for attention blocks (accumulates across layers)
        if data_sizes.kv_cache_bytes > 0:
            # Each layer writes its own KV cache to global memory
            kv_time = self.global_memory.transfer_time_ns(data_sizes.kv_cache_bytes)
            kv_energy = self.global_memory.transfer_energy_nj(data_sizes.kv_cache_bytes)

        return {
            "weight_time": weight_time,
            "weight_energy": weight_energy,
            "comm_time": comm_time,
            "comm_energy": comm_energy,
            "kv_time": kv_time,
            "kv_energy": kv_energy,
            "total_time": weight_time + comm_time + kv_time,
            "total_energy": weight_energy + comm_energy + kv_energy,
        }

    def _compute_all_overheads(self, assignments: Dict[str, DesignPointResult]) -> Dict:
        """Compute overhead for all DFG nodes."""
        G = self.system_dfg
        per_node = {}
        totals = OverheadBreakdown()

        for node_id in nx.topological_sort(G):
            oh = self._compute_node_overhead(node_id, assignments)
            per_node[node_id] = oh
            totals.weight_load_time_ns += oh["weight_time"]
            totals.weight_load_energy_nj += oh["weight_energy"]
            totals.comm_time_ns += oh["comm_time"]
            totals.comm_energy_nj += oh["comm_energy"]
            totals.kv_cache_time_ns += oh["kv_time"]
            totals.kv_cache_energy_nj += oh["kv_energy"]

        return {"totals": totals, "per_node": per_node}

    # ------------------------------------------------------------------
    # System metric composition
    # ------------------------------------------------------------------

    def _compose_system_metrics(
        self,
        assignments: Dict[str, DesignPointResult],
        execution_model: str,
        objective: str,
    ) -> SystemDesignPoint:
        """Compose per-block metrics into system-level metrics including overheads."""
        block_types = self.system_cfg["block_types"]

        # Compute overheads
        overhead_data = self._compute_all_overheads(assignments)
        overhead = overhead_data["totals"]
        per_node_overhead = overhead_data["per_node"]

        # --- Compute-only execution time ---
        if execution_model == "dfg":
            compute_time = self._dfg_critical_path(assignments, per_node_overhead)
        elif execution_model == "pipelined":
            stage_times = []
            for block_name, result in assignments.items():
                stage_times.append(result.execution_time)
            total_repeats = max(
                block_types[bn].get("repeat_count", 1) for bn in assignments
            )
            compute_time = (
                sum(r.execution_time for r in assignments.values())
                + max(stage_times) * (total_repeats - 1)
            )
            # For pipelined, add overhead proportionally
            compute_time += overhead.total_time_ns
        else:  # sequential (default)
            compute_time_only = sum(
                r.execution_time * block_types[bn].get("repeat_count", 1)
                for bn, r in assignments.items()
            )
            # In sequential mode, overhead is added per block instance
            compute_time = compute_time_only + overhead.total_time_ns

        total_time = compute_time

        # --- Energy: compute + overhead ---
        compute_energy = sum(
            (r.total_active_energy + r.total_passive_energy)
            * block_types[bn].get("repeat_count", 1)
            for bn, r in assignments.items()
        )
        total_energy = compute_energy + overhead.total_energy_nj

        # --- Area: sum of unique block areas (dedicated HW per block type) ---
        total_area = sum(r.total_area for r in assignments.values())

        # --- Objective ---
        if objective == "edp":
            obj = total_energy * total_time
        elif objective == "ed2":
            obj = total_energy * total_time ** 2
        elif objective == "delay":
            obj = total_time
        elif objective == "energy":
            obj = total_energy
        else:
            obj = total_energy * total_time  # default to EDP

        return SystemDesignPoint(
            block_assignments=assignments,
            system_execution_time=total_time,
            system_total_energy=total_energy,
            system_total_area=total_area,
            system_obj_value=obj,
            compute_time_ns=compute_time - overhead.total_time_ns,
            compute_energy_nj=compute_energy,
            overhead=overhead,
        )

    def _dfg_critical_path(
        self,
        assignments: Dict[str, DesignPointResult],
        per_node_overhead: Optional[Dict] = None,
    ) -> float:
        """Compute critical path through the system DFG including per-node overheads."""
        G = self.system_dfg
        node_times: Dict[str, float] = {}

        for node in nx.topological_sort(G):
            block_type = G.nodes[node]["block_type"]
            block_time = assignments[block_type].execution_time

            # Add per-node overhead (weight load + comm + kv cache)
            node_overhead = 0.0
            if per_node_overhead and node in per_node_overhead:
                node_overhead = per_node_overhead[node]["total_time"]

            pred_completion = 0.0
            for pred in G.predecessors(node):
                pred_completion = max(pred_completion, node_times[pred])

            node_times[node] = pred_completion + block_time + node_overhead

        return max(node_times.values()) if node_times else 0.0

    def _check_system_constraints(
        self, point: SystemDesignPoint, constraints: dict
    ) -> bool:
        """Check system-level power and area constraints."""
        max_power = float(constraints.get("max_total_power", float("inf")))
        max_area = float(constraints.get("max_total_area", float("inf")))

        if point.system_execution_time > 0:
            total_power = point.system_total_energy / point.system_execution_time
        else:
            total_power = float("inf")

        if total_power > max_power:
            return False
        if point.system_total_area > max_area:
            return False
        return True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, best_system: Optional[SystemDesignPoint]):
        """Save system-level results and produce summary report."""
        report = {
            "system_config": self.system_cfg["name"],
            "execution_model": self.system_cfg.get("execution_model", "sequential"),
            "shared_tech": self.system_cfg.get("shared_tech", False),
            "global_memory": {
                "bandwidth_gbps": self.global_memory.bandwidth_gbps,
                "latency_ns": self.global_memory.latency_ns,
                "energy_per_bit_pj": self.global_memory.energy_per_bit_pj,
            },
            "block_data_sizes": {
                bn: {
                    "input_bytes": ds.input_bytes,
                    "output_bytes": ds.output_bytes,
                    "weight_bytes": ds.weight_bytes,
                    "kv_cache_bytes": ds.kv_cache_bytes,
                    "shared_weights": ds.shared_weights,
                }
                for bn, ds in self.block_data_sizes.items()
            },
            "blocks": {},
        }

        if best_system:
            oh = best_system.overhead or OverheadBreakdown()
            report["system_metrics"] = {
                "execution_time_ns": best_system.system_execution_time,
                "total_energy_nJ": best_system.system_total_energy,
                "total_area_um2": best_system.system_total_area,
                "objective_value": best_system.system_obj_value,
                "compute_time_ns": best_system.compute_time_ns,
                "compute_energy_nJ": best_system.compute_energy_nj,
            }
            report["overhead_breakdown"] = {
                "total_overhead_time_ns": oh.total_time_ns,
                "total_overhead_energy_nJ": oh.total_energy_nj,
                "weight_load_time_ns": oh.weight_load_time_ns,
                "weight_load_energy_nJ": oh.weight_load_energy_nj,
                "comm_time_ns": oh.comm_time_ns,
                "comm_energy_nJ": oh.comm_energy_nj,
                "kv_cache_time_ns": oh.kv_cache_time_ns,
                "kv_cache_energy_nJ": oh.kv_cache_energy_nj,
            }
            for block_name, result in best_system.block_assignments.items():
                repeat = self.system_cfg["block_types"][block_name].get("repeat_count", 1)
                report["blocks"][block_name] = {
                    "repeat_count": repeat,
                    "design_point": result.design_point,
                    "execution_time_ns": result.execution_time,
                    "total_active_energy_nJ": result.total_active_energy,
                    "total_passive_energy_nJ": result.total_passive_energy,
                    "total_area_um2": result.total_area,
                    "clk_period_ns": result.clk_period,
                    "V_dd": result.V_dd,
                    "L": result.L,
                }
        else:
            report["system_metrics"] = None

        report_path = os.path.join(self.save_dir, "system_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"System report saved to {report_path}")

        if best_system:
            pkl_path = os.path.join(self.save_dir, "best_system_design.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(best_system, f)
            logger.info(f"Best system design saved to {pkl_path}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self) -> Optional[SystemDesignPoint]:
        """Run the full system codesign flow.

        If --block-results-dir was provided, Step 1 loads results from a
        previous run instead of invoking codesign subprocesses.
        """
        if self.block_results_dir:
            logger.info("=== Step 1: Load per-block results (checkpoint) ===")
        else:
            logger.info("=== Step 1: Per-block codesign ===")
        self.run_per_block_codesign()

        logger.info("=== Step 2: Build system DFG ===")
        self.system_dfg = self.build_system_dfg()

        # Save DFG to GML for visualization (GML only supports primitives,
        # so serialize edge transfer_config as separate string attributes)
        gml_graph = self.system_dfg.copy()
        for u, v, data in gml_graph.edges(data=True):
            if "transfer_config" in data:
                tc = data.pop("transfer_config")
                data["transfer"] = tc.transfer
                data["on_chip_latency_ns"] = tc.on_chip_latency_ns
                data["on_chip_energy_per_bit_pj"] = tc.on_chip_energy_per_bit_pj
        gml_path = os.path.join(self.save_dir, "system_dfg.gml")
        nx.write_gml(gml_graph, gml_path)
        logger.info(f"System DFG saved to {gml_path}")

        logger.info("=== Step 3: System-level optimization ===")
        best_system = self.optimize_system()

        logger.info("=== Step 4: Report ===")
        self.report(best_system)

        return best_system


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="SystemCodesign",
        description="System-level codesign orchestrator",
    )
    parser.add_argument(
        "--system-config-file",
        type=str,
        default="src/yaml/system_cfg.yaml",
        help="Path to system config YAML file",
    )
    parser.add_argument(
        "--system-config",
        type=str,
        required=True,
        help="Name of the system config to use",
    )
    parser.add_argument(
        "--block-results-dir",
        type=str,
        default=None,
        help="Path to a previous system run directory to load block results from "
             "(skips per-block codesign). E.g. logs/system/2026-02-16_15-47-21_transformer_1layer",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    system = SystemCodesign(
        args.system_config_file, args.system_config,
        block_results_dir=args.block_results_dir,
    )
    best = system.execute()

    if best:
        oh = best.overhead or OverheadBreakdown()
        pct_overhead = (
            100.0 * oh.total_time_ns / best.system_execution_time
            if best.system_execution_time > 0 else 0.0
        )
        print(
            f"\nBest system design:\n"
            f"  Objective:      {best.system_obj_value:.4e}\n"
            f"  Execution time: {best.system_execution_time:.2f} ns\n"
            f"    Compute:      {best.compute_time_ns:.2f} ns\n"
            f"    Overhead:     {oh.total_time_ns:.2f} ns ({pct_overhead:.1f}%)\n"
            f"      Weight load:  {oh.weight_load_time_ns:.2f} ns\n"
            f"      Comm:         {oh.comm_time_ns:.2f} ns\n"
            f"      KV cache:     {oh.kv_cache_time_ns:.2f} ns\n"
            f"  Total energy:   {best.system_total_energy:.4e} nJ\n"
            f"    Compute:      {best.compute_energy_nj:.4e} nJ\n"
            f"    Overhead:     {oh.total_energy_nj:.4e} nJ\n"
            f"  Total area:     {best.system_total_area:.4e} um^2\n"
        )
        for block_name, result in best.block_assignments.items():
            print(f"  Block '{block_name}': V_dd={result.V_dd}, L={result.L}")
    else:
        print("\nNo valid system design found.")


if __name__ == "__main__":
    main()
