#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VF2-based subgraph matching for A2 Trojan template on CDL netlists.

This script parses a flattened/hierarchical CDL netlist, builds an undirected graph where
nodes are net names, and edges indicate co-occurrence within a device instance (transistor/gate).
It then extracts k-hop subgraphs around given abnormal nets and matches them against an A2 template
graph using VF2 (exact) and graph edit distance (approximate).

Typical usage:
    python vf2_a2_match.py \
        --cdl path/to/design.cdl \
        --a2-edge-list path/to/a2_edges.txt \
        --abnormal-nets net1,net2,net3 \
        --num-iterations 2 \
        --connect-mode clique \
        --fuzzy-timeout 2.0
"""

from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
from networkx.algorithms import isomorphism


# -----------------------------
# Parsing: CDL -> device instances
# -----------------------------

_DEVICE_LINE_RE = re.compile(r"^\s*X\S+", re.IGNORECASE)
_COMMENT_RE = re.compile(r"^\s*\*")  # SPICE/CDL comment
_CONTINUATION_RE = re.compile(r"^\s*\+")  # continuation line


@dataclass(frozen=True)
class DeviceInstance:
    """A device instance extracted from CDL.

    Attributes:
        name: Instance name (e.g., XNM1, XU123).
        pins: Ordered pin net names (excluding model name and parameters).
        model: Model/subckt name (best-effort).
        raw: Raw reconstructed line.
    """
    name: str
    pins: Tuple[str, ...]
    model: str
    raw: str


def _merge_continuations(lines: Sequence[str]) -> List[str]:
    """Merge CDL continuation lines starting with '+' into the previous line."""
    merged: List[str] = []
    for line in lines:
        if _COMMENT_RE.match(line) or not line.strip():
            continue
        if _CONTINUATION_RE.match(line):
            if not merged:
                continue
            merged[-1] = merged[-1].rstrip() + " " + line.lstrip("+").strip()
        else:
            merged.append(line.rstrip("\n"))
    return merged


def parse_cdl_instances(
    cdl_path: Path,
    *,
    keep_subckt_calls_only: bool = False,
) -> List[DeviceInstance]:
    """Parse a CDL file and extract device instances (lines starting with 'X').

    Notes:
        - This is a practical parser for open-source reproducibility, not a full CDL grammar.
        - It supports continuation lines and ignores comments.

    Args:
        cdl_path: Path to .cdl netlist.
        keep_subckt_calls_only: If True, keep only subckt calls (still begin with X).

    Returns:
        A list of DeviceInstance objects.
    """
    text = cdl_path.read_text(encoding="utf-8", errors="ignore")
    raw_lines = text.splitlines()
    lines = _merge_continuations(raw_lines)

    instances: List[DeviceInstance] = []
    for line in lines:
        if not _DEVICE_LINE_RE.match(line):
            continue

        # Tokenize: Xname n1 n2 ... model [params...]
        # Heuristic: last "net" ends before model token; model is the first token after nets.
        tokens = line.split()
        if len(tokens) < 3:
            continue

        inst_name = tokens[0]
        # Heuristic to locate model token:
        # We assume model token is the first token that does NOT look like a net assignment "a=b"
        # and does NOT contain '=' and is not a parameter keyword like $PINS.
        # Common CDL style: X.. net net net net <model> <params...>
        # We will find the first token after inst_name that starts with alphabet and has no '='
        # but this is best-effort.
        model_idx = None
        for i in range(2, len(tokens)):
            tok = tokens[i]
            if tok.upper() == "$PINS":
                continue
            if "=" in tok:
                continue
            # likely model/subckt name
            model_idx = i
            break

        if model_idx is None:
            # fallback: treat last token as model, rest as pins
            model_idx = len(tokens) - 1

        pin_tokens = []
        # tokens[1:model_idx] are pins unless there is $PINS style
        for tok in tokens[1:model_idx]:
            if tok.upper() == "$PINS":
                continue
            # handle "A1=net" style
            if "=" in tok:
                _, net = tok.split("=", 1)
                if net:
                    pin_tokens.append(net.strip(","))
            else:
                pin_tokens.append(tok.strip(","))

        model = tokens[model_idx].strip(",")
        if keep_subckt_calls_only and not model:
            continue

        # Clean pins (remove trailing commas, parentheses)
        pins = tuple(p.strip(",()") for p in pin_tokens if p.strip(",()"))
        if len(pins) < 2:
            continue

        instances.append(DeviceInstance(inst_name, pins, model, line))

    return instances


# -----------------------------
# Graph building
# -----------------------------

def build_net_graph(
    instances: Sequence[DeviceInstance],
    *,
    connect_mode: str = "clique",
    drop_power_nets: bool = True,
    power_net_regex: str = r"^(vdd|vss|gnd|vcc|vpwr|vgnd)$",
) -> nx.Graph:
    """Build an undirected net graph from CDL instances.

    Nodes:
        Net names (strings)

    Edges:
        Net co-occurrence inside the same instance.

    Args:
        instances: Parsed device instances.
        connect_mode: 'clique' or 'star'.
        drop_power_nets: Whether to remove obvious power nets.
        power_net_regex: Regex for power net names (case-insensitive).

    Returns:
        A NetworkX undirected graph.
    """
    g = nx.Graph()
    power_re = re.compile(power_net_regex, re.IGNORECASE)

    def is_power(net: str) -> bool:
        return bool(power_re.match(net))

    for inst in instances:
        pins = list(inst.pins)

        if drop_power_nets:
            pins = [p for p in pins if not is_power(p)]
        if len(pins) < 2:
            continue

        # add nodes
        for p in pins:
            g.add_node(p)

        # add edges
        if connect_mode == "clique":
            for u, v in itertools.combinations(pins, 2):
                if u != v:
                    g.add_edge(u, v)
        elif connect_mode == "star":
            center = pins[0]
            for v in pins[1:]:
                if center != v:
                    g.add_edge(center, v)
        else:
            raise ValueError(f"Unknown connect_mode: {connect_mode}")

    return g


# -----------------------------
# Subgraph extraction (BFS hops)
# -----------------------------

def k_hop_subgraph_nx(
    g: nx.Graph,
    center: str,
    k: int,
) -> nx.Graph:
    """Extract k-hop induced subgraph around a center node using BFS."""
    if center not in g:
        return nx.Graph()

    visited: Set[str] = {center}
    frontier: Set[str] = {center}

    for _ in range(k):
        next_frontier: Set[str] = set()
        for u in frontier:
            next_frontier.update(g.neighbors(u))
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    return g.subgraph(visited).copy()


# -----------------------------
# Matching: VF2 (exact) + GED (fuzzy)
# -----------------------------

def exact_subgraph_match_percentage(main_g: nx.Graph, pattern_g: nx.Graph) -> float:
    """Exact VF2 subgraph match. Returns 100.0 if matched, else 0.0."""
    if pattern_g.number_of_nodes() == 0 or pattern_g.number_of_edges() == 0:
        return 0.0
    gm = isomorphism.GraphMatcher(main_g, pattern_g)
    return 100.0 if gm.subgraph_is_isomorphic() else 0.0


def fuzzy_match_percentage_by_ged(
    main_g: nx.Graph,
    pattern_g: nx.Graph,
    *,
    timeout: float = 2.0,
    upper_bound: Optional[float] = None,
) -> float:
    """Approximate matching using (normalized) graph edit distance (GED).

    We convert GED to similarity:
        sim = 1 - ged / max(|E_main|, |E_pat|, 1)
        percentage = sim * 100

    Args:
        main_g: Candidate subgraph.
        pattern_g: A2 pattern graph.
        timeout: Max seconds for GED computation.
        upper_bound: Optional upper bound for GED (pruning).

    Returns:
        Similarity percentage in [0, 100].
    """
    if pattern_g.number_of_nodes() == 0:
        return 0.0

    # NetworkX GED can be expensive; we rely on timeout.
    # If it returns None, treat as 0 similarity.
    ged = nx.graph_edit_distance(
        main_g,
        pattern_g,
        timeout=timeout,
        upper_bound=upper_bound,
    )
    if ged is None:
        return 0.0

    denom = max(main_g.number_of_edges(), pattern_g.number_of_edges(), 1)
    sim = max(0.0, 1.0 - float(ged) / float(denom))
    return 100.0 * sim


def load_edge_list(path: Path) -> nx.Graph:
    """Load an undirected graph from an edge list file.

    File format:
        each line: <u> <v>
        lines starting with # are ignored

    Node labels can be net names or integers; they are kept as strings to be consistent.
    """
    g = nx.Graph()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        u, v, *rest = s.split()
        g.add_edge(u, v)
    return g

def load_abnormal_nets(jsonl_path: Path) -> List[str]:
    """Load abnormal nets from a jsonl file.

    Supported formats:
        1) {"abnormal_nets": ["n1", "n2", ...]}
        2) {"abnormal_net": "n1"}

    Returns:
        A list of abnormal net names.
    """
    nets: List[str] = []
    for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        if "abnormal_nets" in obj and isinstance(obj["abnormal_nets"], list):
            nets.extend([str(x) for x in obj["abnormal_nets"]])
        elif "abnormal_net" in obj:
            nets.append(str(obj["abnormal_net"]))
    # de-dup while preserving order
    seen = set()
    out = []
    for n in nets:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


# -----------------------------
# Main pipeline
# -----------------------------

def run_matching(
    design_g: nx.Graph,
    a2_g: nx.Graph,
    abnormal_nets: Sequence[str],
    *,
    k_hops: int,
    fuzzy: bool,
    fuzzy_timeout: float,
    fuzzy_upper_bound: Optional[float],
) -> List[Tuple[str, float]]:
    """Run subgraph extraction + matching for each abnormal net."""
    results: List[Tuple[str, float]] = []

    for net in abnormal_nets:
        sub_g = k_hop_subgraph_nx(design_g, net, k_hops)
        if sub_g.number_of_nodes() == 0:
            results.append((net, 0.0))
            continue

        exact = exact_subgraph_match_percentage(sub_g, a2_g)
        if exact >= 100.0:
            results.append((net, 100.0))
            continue

        if fuzzy:
            score = fuzzy_match_percentage_by_ged(
                sub_g,
                a2_g,
                timeout=fuzzy_timeout,
                upper_bound=fuzzy_upper_bound,
            )
            results.append((net, score))
        else:
            results.append((net, 0.0))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VF2 subgraph matching for A2 Trojan template on CDL netlists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--abnormal-jsonl", type=Path, default=None,
                        help="Path to anomalies.jsonl produced by IsolationForest screening.")
    parser.add_argument("--cdl", type=Path, required=True, help="Path to the input .cdl netlist.")
    parser.add_argument("--a2-edge-list", type=Path, required=True,
                        help="Path to A2 template edge list file.")
    parser.add_argument("--abnormal-nets", type=str, required=True,
                        help="Comma-separated abnormal net names, e.g., n1,n2,n3")
    parser.add_argument("--k-hops", type=int, default=2,
                        help="k-hop radius for subgraph extraction.")
    parser.add_argument("--connect-mode", type=str, default="clique", choices=["clique", "star"],
                        help="How to connect nets within one device instance.")
    parser.add_argument("--keep-power-nets", action="store_true",
                        help="If set, keep power nets (VDD/VSS/GND/etc.) in graph.")
    parser.add_argument("--power-net-regex", type=str,
                        default=r"^(vdd|vss|gnd|vcc|vpwr|vgnd)$",
                        help="Regex for power net names.")
    parser.add_argument("--fuzzy", action="store_true",
                        help="Enable fuzzy matching by graph edit distance.")
    parser.add_argument("--fuzzy-timeout", type=float, default=2.0,
                        help="Timeout (seconds) for GED computation per net.")
    parser.add_argument("--fuzzy-upper-bound", type=float, default=None,
                        help="Optional upper bound for GED to prune search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Parse CDL and build design graph
    instances = parse_cdl_instances(args.cdl)
    if not instances:
        raise RuntimeError(f"No device instances parsed from CDL: {args.cdl}")

    design_g = build_net_graph(
        instances,
        connect_mode=args.connect_mode,
        drop_power_nets=not args.keep_power_nets,
        power_net_regex=args.power_net_regex,
    )
    if design_g.number_of_nodes() == 0:
        raise RuntimeError("Empty design graph. Check CDL parsing rules or netlist format.")

    # 2) Load A2 template graph
    a2_g = load_edge_list(args.a2_edge_list)
    if a2_g.number_of_nodes() == 0:
        raise RuntimeError(f"Empty A2 template graph: {args.a2_edge_list}")

    # 3) Load abnormal nets (jsonl has higher priority)
    if args.abnormal_jsonl is not None:
        if not args.abnormal_jsonl.exists():
            raise FileNotFoundError(f"--abnormal-jsonl not found: {args.abnormal_jsonl}")
        abnormal_nets = load_abnormal_nets(args.abnormal_jsonl)
    elif args.abnormal_nets is not None:
        abnormal_nets = [x.strip() for x in args.abnormal_nets.split(",") if x.strip()]
    else:
        raise ValueError("You must provide either --abnormal-jsonl or --abnormal-nets.")

    if not abnormal_nets:
        raise ValueError("No abnormal nets found. Check anomalies.jsonl or --abnormal-nets input.")

    # Optional: filter abnormal nets that are not in the design graph
    missing = [n for n in abnormal_nets if n not in design_g]
    if missing:
        print(f"[WARN] {len(missing)} abnormal nets are not in the design graph. "
              f"Example: {missing[:10]}")
        abnormal_nets = [n for n in abnormal_nets if n in design_g]

    if not abnormal_nets:
        raise ValueError("All abnormal nets are missing from the design graph. "
                         "Your anomaly output may be using different naming than the CDL netlist.")

    # 4) Run matching
    results = run_matching(
        design_g=design_g,
        a2_g=a2_g,
        abnormal_nets=abnormal_nets,
        k_hops=args.k_hops,
        fuzzy=args.fuzzy,
        fuzzy_timeout=args.fuzzy_timeout,
        fuzzy_upper_bound=args.fuzzy_upper_bound,
    )

    # 5) Print results
    print("\n=== Matching Results ===")
    for net, score in results:
        print(f"[abnormal_net={net}] match={score:.2f}%")

    # 6) Summary stats (useful for open-source reproducibility)
    print("\n=== Summary ===")
    print(f"CDL: {args.cdl}")
    print(f"A2 template edge list: {args.a2_edge_list}")
    if args.abnormal_jsonl is not None:
        print(f"Abnormal source: {args.abnormal_jsonl}  (count={len(abnormal_nets)})")
    else:
        print(f"Abnormal source: --abnormal-nets  (count={len(abnormal_nets)})")

    print(f"Design graph: |V|={design_g.number_of_nodes()} |E|={design_g.number_of_edges()}")
    print(f"A2 template : |V|={a2_g.number_of_nodes()} |E|={a2_g.number_of_edges()}")
    print(f"Params: k_hops={args.k_hops}, connect_mode={args.connect_mode}, "
          f"keep_power_nets={args.keep_power_nets}, fuzzy={args.fuzzy}, "
          f"fuzzy_timeout={args.fuzzy_timeout}, fuzzy_upper_bound={args.fuzzy_upper_bound}")



if __name__ == "__main__":
    main()
