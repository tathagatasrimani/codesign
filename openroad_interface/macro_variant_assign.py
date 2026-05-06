"""Stage 2: map each FU/mux node to a macro height variant (__lo / base / __hi) from LEF.

MacroMaker emits three LEF cells per op when enough valid row-height designs exist; the
middle cell keeps the base name (e.g. Add16), the others use ``__lo`` / ``__hi`` suffixes.

Disable reassignment (keep all instances on the base/mid cell) with YAML::

    args:
      use_macro_variants: false

Optional tuning for region-aware recursion:

    args:
      macro_variant_region_leaf_size: 6
      macro_variant_region_blend: 0.8
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


def _variants_for_base(
    base: str, macro_size_dict: Dict[str, Tuple[float, float]]
) -> List[Tuple[str, float]]:
    """Return [(lef_name, height/width), ...] sorted by aspect for one logical base name."""
    pairs: List[Tuple[str, float]] = []
    for k, wh in macro_size_dict.items():
        if k == base or (k.startswith(base + "__") and k != base):
            sx, sy = wh
            if sx <= 0:
                continue
            pairs.append((k, sy / sx))
    pairs.sort(key=lambda t: t[1])
    return pairs


def _lef_has_height_variants(macro_size_dict: Dict[str, Any]) -> bool:
    return any(k.endswith("__lo") or k.endswith("__hi") for k in macro_size_dict)


def _node_area_from_macro(
    node: str,
    node_to_macro: Dict[str, List[Any]],
    macro_size_dict: Dict[str, Tuple[float, float]],
) -> float:
    base = node_to_macro[node][0]
    if base in macro_size_dict:
        w, h = macro_size_dict[base]
        return max(0.0, w * h)
    return 0.0


def _recursive_region_aspect_targets(
    ordered: List[str],
    node_to_macro: Dict[str, List[Any]],
    macro_size_dict: Dict[str, Tuple[float, float]],
    root_aspect_hw: float,
    leaf_size: int,
) -> Dict[str, float]:
    """
    Approximate recursive strip partitioning and return per-node region aspect targets.

    Mirrors the placer's alternating strip orientation:
    even depth -> horizontal strips (height split), odd depth -> vertical strips (width split).
    """
    out: Dict[str, float] = {}
    min_frac = 0.15

    def rec(bucket: List[str], aspect_hw: float, depth: int) -> None:
        if not bucket:
            return
        a = max(1e-3, float(aspect_hw))
        if len(bucket) <= leaf_size:
            for n in bucket:
                out[n] = a
            return

        total = sum(_node_area_from_macro(n, node_to_macro, macro_size_dict) for n in bucket)
        if total <= 1e-12:
            mid = len(bucket) // 2
            left = bucket[:mid]
            right = bucket[mid:]
            frac = 0.5
        else:
            acc = 0.0
            split_idx = 1
            target = 0.5 * total
            for i, n in enumerate(bucket[:-1], start=1):
                acc += _node_area_from_macro(n, node_to_macro, macro_size_dict)
                split_idx = i
                if acc >= target:
                    break
            left = bucket[:split_idx]
            right = bucket[split_idx:]
            frac = max(min_frac, min(1.0 - min_frac, acc / total))

        if not left or not right:
            for n in bucket:
                out[n] = a
            return

        if depth % 2 == 0:
            # Horizontal strip split: width fixed, height scales by fraction.
            a_left = a * frac
            a_right = a * (1.0 - frac)
        else:
            # Vertical strip split: height fixed, width scales by fraction.
            a_left = a / max(frac, 1e-6)
            a_right = a / max(1.0 - frac, 1e-6)

        rec(left, a_left, depth + 1)
        rec(right, a_right, depth + 1)

    rec(ordered, max(1e-3, root_aspect_hw), 0)
    return out


def assign_macro_variants(
    old_graph: nx.DiGraph,
    node_to_macro: Dict[str, List[Any]],
    macro_dict: Dict[str, Any],
    macro_size_dict: Dict[str, Tuple[float, float]],
    core_aspect_hw: Optional[float],
    cfg: Optional[dict] = None,
) -> None:
    """
    Mutate node_to_macro[node][0] and [...][1] to pick Add16__lo / Add16 / Add16__hi etc.
    Skips nodes whose base macro has only one LEF entry (e.g. Register, hierarchical).
    """
    if isinstance(cfg, dict):
        args = cfg.get("args")
        if isinstance(args, dict) and args.get("use_macro_variants") is False:
            logger.info("Macro variant assignment skipped (args.use_macro_variants is false).")
            return
    if not _lef_has_height_variants(macro_size_dict):
        return

    try:
        topo = list(nx.topological_sort(old_graph))
    except nx.NetworkXUnfeasible:
        topo = sorted(old_graph.nodes())

    ordered = [n for n in topo if n in node_to_macro]
    if not ordered:
        return

    core_a = core_aspect_hw if core_aspect_hw is not None and core_aspect_hw > 0 else 1.0
    leaf_size = 6
    blend = 0.8
    if isinstance(cfg, dict):
        args = cfg.get("args")
        if isinstance(args, dict):
            if args.get("macro_variant_region_leaf_size") is not None:
                leaf_size = max(2, int(args.get("macro_variant_region_leaf_size")))
            if args.get("macro_variant_region_blend") is not None:
                blend = float(args.get("macro_variant_region_blend"))
    blend = max(0.0, min(1.0, blend))

    node_target_aspect = _recursive_region_aspect_targets(
        ordered, node_to_macro, macro_size_dict, core_a, leaf_size
    )
    n_assigned = 0

    for node in ordered:
        base = node_to_macro[node][0]
        variants = _variants_for_base(base, macro_size_dict)
        if len(variants) < 2:
            continue
        region_a = node_target_aspect.get(node, core_a)
        goal = blend * region_a + (1.0 - blend) * core_a
        best_name, _best_a = min(variants, key=lambda va: abs(va[1] - goal))
        if best_name == node_to_macro[node][0]:
            continue
        if best_name not in macro_dict:
            logger.warning(
                "Variant %s not in macro_dict; skipping node %s", best_name, node
            )
            continue
        sub = copy.deepcopy(macro_dict[best_name])
        fn = None
        if base in macro_dict:
            fn = macro_dict[base].get("function")
        if not fn:
            fn = old_graph.nodes[node].get("function", "")
        sub["function"] = fn
        node_to_macro[node][0] = best_name
        node_to_macro[node][1] = sub
        n_assigned += 1

    if n_assigned:
        logger.info(
            "Macro variant assignment (stage 2, region-aware recursive): "
            "reassigned %d of %d FU/mux nodes.",
            n_assigned,
            len(ordered),
        )
