"""Utilities to estimate the empirical time complexity of the latent prefilter.

This script loads the hierarchical clustering tree that powers the prefilter
and simulates a best-first traversal where the largest remaining cluster is
expanded until at least ``N`` sequences (``size``) worth of clusters have been
materialised.  The sum of sequence counts under the expanded leaves represents
``N`` as described in the request.

The script only needs a subset of the original tree and allows restricting the
analysis to an arbitrary subtree or to a maximum depth.  Nodes at or beyond the
cut depth are treated as aggregate leaves whose ``size`` already captures the
number of descendant sequences, so the traversal still provides a faithful
account of the amount of work required before touching deeper parts of the
hierarchy.

Example
-------
$ python usage/prefilter_complexity.py --targets 10 25 50 --subtree-root 750 \\
      --max-depth 4

This will report how many heap operations, node expansions, and leaf material-
isation steps are necessary to accumulate at least 10, 25, and 50 sequences
using only the subtree rooted at node 750 up to depth four.
"""

from __future__ import annotations

import argparse
import heapq
import json
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TreeNode:
    """Compact representation for a node in the clustering tree."""

    node_id: int
    size: int
    is_leaf: bool
    children: Tuple[int, ...]
    depth: int


def _load_nodes(index_dir: Path) -> pd.DataFrame:
    nodes_path = index_dir / "nodes.parquet"
    if not nodes_path.exists():
        raise FileNotFoundError(
            f"Cannot find {nodes_path}. Ensure the clustering index is available."
        )
    return pd.read_parquet(nodes_path)


def _build_subtree(
    nodes_df: pd.DataFrame, *, root_id: int, max_depth: Optional[int]
) -> Dict[int, TreeNode]:
    """Extract a subtree and optionally truncate it at ``max_depth``.

    Parameters
    ----------
    nodes_df:
        DataFrame containing the raw node description.
    root_id:
        Identifier of the node that will act as the root of the subtree.
    max_depth:
        Optional depth (relative to ``root_id``) where the tree should be cut.
        Nodes at the cut depth are treated as leaves.
    """

    lookup = nodes_df.set_index("node_id")
    if root_id not in lookup.index:
        raise KeyError(f"Node {root_id} is not present in the clustering index")

    subtree: Dict[int, TreeNode] = {}
    queue: List[Tuple[int, int]] = [(root_id, 0)]
    while queue:
        node_id, depth = queue.pop()
        if node_id in subtree:
            continue
        row = lookup.loc[node_id]
        children: List[int] = []
        if not row["is_leaf"]:
            if row["left_id"] != -1:
                children.append(int(row["left_id"]))
            if row["right_id"] != -1:
                children.append(int(row["right_id"]))

        truncate_here = max_depth is not None and depth >= max_depth
        is_leaf = bool(row["is_leaf"]) or truncate_here or not children
        if not truncate_here:
            for child_id in children:
                queue.append((child_id, depth + 1))
        else:
            children = []

        subtree[node_id] = TreeNode(
            node_id=int(node_id),
            size=int(row["size"]),
            is_leaf=is_leaf,
            children=tuple(children),
            depth=depth,
        )

    return subtree


def _simulate_prefilter(
    tree: Dict[int, TreeNode],
    *,
    root_id: int,
    target_sequences: int,
) -> Dict[str, float]:
    """Simulate the best-first traversal used by the latent prefilter.

    The simulation expands the node that currently covers the largest number of
    sequences until the accumulated number of sequences across leaf nodes
    reaches ``target_sequences``.
    """

    if target_sequences <= 0:
        raise ValueError("target_sequences must be positive")

    if root_id not in tree:
        raise KeyError(f"Root {root_id} is not present in the provided subtree")

    root_node = tree[root_id]
    capped_target = min(target_sequences, root_node.size)

    heap: List[Tuple[int, int]] = [(-root_node.size, root_node.node_id)]
    heapq.heapify(heap)

    popped_nodes = 0
    expanded_internal = 0
    materialised_leaves = 0
    sequences_accumulated = 0
    popped_sequence_total = 0

    start = time.perf_counter()


    while heap and sequences_accumulated < capped_target:
        neg_size, node_id = heapq.heappop(heap)
        popped_nodes += 1
        node = tree[node_id]
        node_size = -neg_size
        popped_sequence_total += node_size


        if node.is_leaf or not node.children:
            sequences_accumulated += node.size
            materialised_leaves += 1
            continue

        expanded_internal += 1
        for child_id in node.children:
            child = tree[child_id]
            heapq.heappush(heap, (-child.size, child.node_id))

    overshoot = (
        float(sequences_accumulated) / float(capped_target)
        if capped_target > 0
        else 0.0
    )
    elapsed = time.perf_counter() - start


    return {
        "target_sequences": float(target_sequences),
        "effective_target": float(capped_target),
        "popped_nodes": float(popped_nodes),
        "expanded_internal": float(expanded_internal),
        "materialised_leaves": float(materialised_leaves),
        "sequences_accumulated": float(sequences_accumulated),
        "popped_sequence_total": float(popped_sequence_total),
        "log_target": float(np.log(max(capped_target, 1))),
        "overshoot": overshoot,
        "elapsed_seconds": float(elapsed),

    }


def _parse_targets(
    requested_targets: Optional[Iterable[int]],
    *,
    max_sequences: int,
    num_targets: int,
) -> List[int]:
    if requested_targets:
        targets = sorted({int(t) for t in requested_targets if int(t) > 0})
        if not targets:
            raise ValueError("At least one positive target must be provided")
        return targets

    # Generate logarithmically spaced targets by default.
    max_sequences = max(max_sequences, 1)
    space = np.geomspace(1, max_sequences, num=num_targets)
    unique_targets = sorted({int(round(x)) for x in space})
    return [t for t in unique_targets if t > 0]


def run_analysis(args: argparse.Namespace) -> Dict[str, object]:
    index_dir = Path(args.index_dir)
    nodes_df = _load_nodes(index_dir)
    subtree = _build_subtree(nodes_df, root_id=args.subtree_root, max_depth=args.max_depth)

    root = subtree[args.subtree_root]
    targets = _parse_targets(args.targets, max_sequences=root.size, num_targets=args.num_targets)

    analysis_start = time.perf_counter()
    results = [_simulate_prefilter(subtree, root_id=root.node_id, target_sequences=t) for t in targets]
    analysis_elapsed = time.perf_counter() - analysis_start

    cumulative_sequence_total = float(sum(r["popped_sequence_total"] for r in results))
    cumulative_accumulated_sequences = float(sum(r["sequences_accumulated"] for r in results))


    return {
        "index_dir": str(index_dir),
        "subtree_root": int(root.node_id),
        "max_depth": args.max_depth,
        "root_size": int(root.size),
        "targets": targets,
        "results": results,
        "analysis_elapsed_seconds": float(analysis_elapsed),
        "cumulative_popped_sequence_total": cumulative_sequence_total,
        "cumulative_sequences_accumulated": cumulative_accumulated_sequences,

    }


def _print_table(results: List[Dict[str, float]]) -> None:
    if not results:
        print("No results to display")
        return

    header = (
        "target",
        "effective",
        "popped",
        "expanded",
        "leaves",
        "accumulated",
        "popped_sum",
        "overshoot",
        "elapsed_s",

    )
    print("\t".join(header))
    for row in results:
        formatted = []
        for key in (
            "target_sequences",
            "effective_target",
            "popped_nodes",
            "expanded_internal",
            "materialised_leaves",
            "sequences_accumulated",
            "popped_sequence_total",
            "overshoot",
            "elapsed_seconds",

        ):
            value = row[key]
            if key == "overshoot":
                formatted.append(f"{value:.2f}")
            elif key == "elapsed_seconds":
                formatted.append(f"{value:.3f}")

            else:
                formatted.append(f"{value:.0f}")
        print("\t".join(formatted))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        default="hclust/index",
        help="Path to the directory containing nodes.parquet (default: hclust/index)",
    )
    parser.add_argument(
        "--subtree-root",
        type=int,
        default=874,
        help="Identifier of the node that acts as the root for the analysis",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum depth (relative to the subtree root) to analyse",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="*",
        help="Explicit target sequence counts to evaluate",
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=6,
        help="Number of logarithmically spaced targets to generate when --targets is omitted",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the results as JSON instead of a human-readable table",
    )

    args = parser.parse_args()
    summary = run_analysis(args)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Subtree root: {summary['subtree_root']} (size={summary['root_size']}) | "
            f"max_depth={summary['max_depth']} | index_dir={summary['index_dir']}"
        )
        _print_table(summary["results"])


if __name__ == "__main__":
    main()
