#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isolation Forest screening on CDL netlists.

This script parses a CDL netlist, builds a net graph (nodes are net names, edges indicate
co-occurrence within a device instance), then derives simple structural features per net node.
An Isolation Forest is trained to identify anomalous nets as candidates for subsequent
VF2 subgraph matching.

Outputs:
    - anomalies.jsonl: abnormal net(s) for VF2 expansion centers.

Typical usage:
    python isoforest_screen.py \
        --cdl ./demo/design.cdl \
        --out ./demo/anomalies.jsonl \
        --connect-mode clique \
        --contamination 0.10 \
        --topk 50
"""

from __future__ import annotations

import argparse
import json
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest


# -----------------------------
# Parsing: CDL -> instances
# -----------------------------

_DEVICE_LINE_RE = re.compile(r"^\s*X\S+", re.IGNORECASE)
_COMMENT_RE = re.compile(r"^\s*\*")
_CONTINUATION_RE = re.compile(r"^\s*\+")


@dataclass(frozen=True)
class DeviceInstance:
    """A device instance parsed from CDL."""
    name: str
    pins: Tuple[str, ...]
    model: str
    raw: str


def _merge_continuations(lines: Sequence[str]) -> List[str]:
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


def parse_cdl_instances(cdl_path: Path) -> List[DeviceInstance]:
    """Extract 'X...' instance lines from a CDL netlist (best-effort)."""
    text = cdl_path.read_text(encoding="utf-8", errors="ignore")
    lines = _merge_continuations(text.splitlines())

    instances: List[DeviceInstance] = []
    for line in lines:
        if not _DEVICE_LINE_RE.match(line):
            continue

        tokens = line.split()
        if len(tokens) < 3:
            continue

        inst_name = tokens[0]

        # Heuristic: locate model token (first token without '=' after pins region)
        model_idx = None
        for i in range(2, len(tokens)):
            tok = tokens[i]
            if tok.upper() == "$PINS":
                continue
            if "=" in tok:
                continue
            model_idx = i
            break
        if model_idx is None:
            model_idx = len(tokens) - 1

        pin_tokens: List[str] = []
        for tok in tokens[1:model_idx]:
            if tok.upper() == "$PINS":
                continue
            if "=" in tok:
                _, net = tok.split("=", 1)
                net = net.strip(",()")
                if net:
                    pin_tokens.append(net)
            else:
                pin_tokens.append(tok.strip(",()"))

        model = tokens[model_idx].strip(",()")
        pins = tuple(p for p in pin_tokens if p)

        if len(pins) < 2:
            continue

        instances.append(DeviceInstance(inst_name, pins, model, line))

    return instances


# -----------------------------
# Graph building: nets as nodes
# -----------------------------

def build_net_graph(
    instances: Sequence[DeviceInstance],
    *,
    connect_mode: str = "clique",
    drop_power_nets: bool = True,
    power_net_regex: str = r"^(vdd|vss|gnd|vcc|vpwr|vgnd)$",
) -> nx.Graph:
    """Build an undirected graph where nodes are net names."""
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

        for p in pins:
            g.add_node(p)

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
# Feature engineering (net-level)
# -----------------------------

def compute_net_features(g: nx.Graph, nodes: List[str]) -> np.ndarray:
    """Compute simple structural features for each net node.

    Features (per node):
        1) degree
        2) clustering coefficient
        3) average neighbor degree
        4) ego network edge count (1-hop induced subgraph edges)
        5) ego network node count (1-hop induced subgraph nodes)

    Returns:
        X: shape (N, 5)
    """
    deg = dict(g.degree())
    clust = nx.clustering(g)
    avg_nbr_deg = nx.average_neighbor_degree(g)

    X = np.zeros((len(nodes), 5), dtype=np.float32)
    for i, n in enumerate(nodes):
        ego = nx.ego_graph(g, n, radius=1, center=True, undirected=True)
        X[i, 0] = float(deg.get(n, 0))
        X[i, 1] = float(clust.get(n, 0.0))
        X[i, 2] = float(avg_nbr_deg.get(n, 0.0))
        X[i, 3] = float(ego.number_of_edges())
        X[i, 4] = float(ego.number_of_nodes())
    return X


# -----------------------------
# Output
# -----------------------------

def write_anomalies_jsonl(out_path: Path, abnormal_nets: List[str]) -> None:
    """Write anomalies for VF2 as jsonl."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"abnormal_nets": abnormal_nets}
    out_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Isolation Forest screening on CDL netlists (net-level graph features).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cdl", type=Path, required=True, help="Input CDL netlist path.")
    p.add_argument("--out", type=Path, default=Path("anomalies.jsonl"), help="Output jsonl path.")
    p.add_argument("--connect-mode", type=str, default="clique", choices=["clique", "star"],
                   help="How to connect nets within a device instance.")
    p.add_argument("--keep-power-nets", action="store_true", help="Keep power nets in graph.")
    p.add_argument("--power-net-regex", type=str, default=r"^(vdd|vss|gnd|vcc|vpwr|vgnd)$",
                   help="Regex for power net names.")
    p.add_argument("--contamination", type=float, default=0.10,
                   help="Expected fraction of anomalies in IsolationForest.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--topk", type=int, default=50,
                   help="Output top-k most anomalous nets (by decision score).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    instances = parse_cdl_instances(args.cdl)
    if not instances:
        raise RuntimeError(f"No device instances parsed from CDL: {args.cdl}")

    g = build_net_graph(
        instances,
        connect_mode=args.connect_mode,
        drop_power_nets=not args.keep_power_nets,
        power_net_regex=args.power_net_regex,
    )
    if g.number_of_nodes() == 0:
        raise RuntimeError("Empty net graph; check parsing rules or netlist format.")

    nodes = sorted(g.nodes())
    X = compute_net_features(g, nodes)

    # Train IsolationForest on all nodes (unsupervised)
    iso = IsolationForest(
        n_estimators=200,
        contamination=args.contamination,
        random_state=args.random_state,
        n_jobs=-1,
    )
    iso.fit(X)

    # decision_function: higher = more normal, lower = more abnormal
    scores = iso.decision_function(X)  # shape (N,)
    # Sort by increasing score => most abnormal first
    rank = np.argsort(scores)

    topk = min(args.topk, len(nodes))
    abnormal_nodes = [nodes[i] for i in rank[:topk]]

    print("=== IsolationForest Screening ===")
    print(f"CDL: {args.cdl}")
    print(f"Graph: |V|={g.number_of_nodes()} |E|={g.number_of_edges()}")
    print(f"Contamination={args.contamination}, topk={topk}")
    print("Top anomalous nets:")
    for i, net in enumerate(abnormal_nodes[:min(20, len(abnormal_nodes))], 1):
        print(f"  {i:02d}. {net}  score={scores[nodes.index(net)]:.6f}")

    write_anomalies_jsonl(args.out, abnormal_nodes)
    print(f"\nSaved anomalies to: {args.out}")


if __name__ == "__main__":
    main()
