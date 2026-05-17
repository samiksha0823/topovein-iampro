"""
TopoVein — File 6: Skeleton Graph Extraction
============================================
Task: Convert skeletonized finger-vein images into real NetworkX graphs.

Why this step matters for research:
    Skeleton images are still pixel grids. For topology-based biometrics and
    downstream persistent homology, we need a graph G(V, E) where:

    - Nodes represent meaningful anatomical events in the skeleton:
      endpoints, bifurcations, crossings, and isolated vein fragments.
    - Edges represent vessel branches between those nodes.
    - Edge weights preserve geometry such as geodesic branch length.

    This script turns each 1-pixel-wide skeleton into a compressed
    `networkx.MultiGraph` so we do not lose:

    - parallel vessel paths between the same two nodes
    - self-loops created by cyclic vein structures
    - branch-level geometry needed later for matching / TDA

Input:
    results/skeleton/<session>/<subject_finger>/<image>_binary.png

Output:
    results/graphs/<session>/<subject_finger>/<image>_graph.pkl
        Pickled `networkx.MultiGraph`

    results/graph_summary.csv
        Per-image graph statistics for documentation and analysis

    results/06_graph_log.txt
        Run log for batch processing

Usage:
    python src/graph_builder/06_graph.py
    python src/graph_builder/06_graph.py --limit 20
    python src/graph_builder/06_graph.py --overwrite --min-component-pixels 4
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import networkx as nx
import numpy as np


Pixel = Tuple[int, int]  # (row, col)


SKELETON_ROOT = Path("results/skeleton")
GRAPH_ROOT = Path("results/graphs")
SUMMARY_CSV = Path("results/graph_summary.csv")
LOG_FILE = Path("results/06_graph_log.txt")

SKIP_EXISTING = True
MIN_COMPONENT_PIXELS = 8
MIN_SPUR_LENGTH = 0.0

NEIGHBOR_OFFSETS: Sequence[Pixel] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


@dataclass(frozen=True)
class ExtractionConfig:
    skeleton_root: Path = SKELETON_ROOT
    graph_root: Path = GRAPH_ROOT
    summary_csv: Path = SUMMARY_CSV
    log_file: Path = LOG_FILE
    skip_existing: bool = SKIP_EXISTING
    min_component_pixels: int = MIN_COMPONENT_PIXELS
    min_spur_length: float = MIN_SPUR_LENGTH
    store_edge_pixels: bool = True


def segment_key(a: Pixel, b: Pixel) -> Tuple[Pixel, Pixel]:
    """Canonical key for an undirected skeleton segment."""
    return (a, b) if a <= b else (b, a)


def step_length(a: Pixel, b: Pixel) -> float:
    """Euclidean length between two 8-connected pixels."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def choose_representative_pixel(pixels: Sequence[Pixel]) -> Tuple[Pixel, float, float]:
    """Return a stable representative pixel and the centroid of a pixel set."""
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    centroid_row = float(np.mean(rows))
    centroid_col = float(np.mean(cols))
    representative = min(
        pixels,
        key=lambda p: ((p[0] - centroid_row) ** 2 + (p[1] - centroid_col) ** 2, p),
    )
    return representative, centroid_row, centroid_col


def compute_output_path(skeleton_path: Path, skeleton_root: Path, graph_root: Path) -> Path:
    """Map a skeleton image path to its graph pickle output path."""
    rel_path = skeleton_path.relative_to(skeleton_root)
    stem = rel_path.stem.replace("_binary", "")
    out_dir = graph_root / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_graph.pkl"


def load_skeleton(path: Path) -> Optional[np.ndarray]:
    """Load a skeleton image as a boolean array."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image > 0


def prune_small_components(
    skeleton: np.ndarray,
    min_component_pixels: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Remove tiny disconnected skeleton fragments that are almost always noise.

    Returns:
        cleaned_skeleton, removed_component_count, removed_pixel_count
    """
    if min_component_pixels <= 1 or not np.any(skeleton):
        return skeleton, 0, 0

    binary = skeleton.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(skeleton, dtype=bool)
    removed_components = 0
    removed_pixels = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_pixels:
            cleaned[labels == label] = True
        else:
            removed_components += 1
            removed_pixels += area

    return cleaned, removed_components, removed_pixels


def build_neighbor_map(skeleton: np.ndarray) -> Dict[Pixel, List[Pixel]]:
    """Create an 8-neighborhood map for all foreground skeleton pixels."""
    pixels = [tuple(map(int, rc)) for rc in np.argwhere(skeleton)]
    pixel_set = set(pixels)
    neighbors: Dict[Pixel, List[Pixel]] = {}

    for row, col in pixels:
        local_neighbors: List[Pixel] = []
        for d_row, d_col in NEIGHBOR_OFFSETS:
            candidate = (row + d_row, col + d_col)
            if candidate in pixel_set:
                local_neighbors.append(candidate)
        neighbors[(row, col)] = sorted(local_neighbors)

    return neighbors


def trace_branch(
    start_pixel: Pixel,
    first_pixel: Pixel,
    pixel_to_node: Dict[Pixel, int],
    neighbor_map: Dict[Pixel, List[Pixel]],
    visited_segments: set,
) -> Tuple[Pixel, List[Pixel], float, str]:
    """
    Trace one branch until it reaches another node or an unexpected terminal.

    Returns:
        terminal_pixel, path_pixels, geodesic_length, status
    """
    path = [start_pixel, first_pixel]
    geodesic_length = step_length(start_pixel, first_pixel)
    visited_segments.add(segment_key(start_pixel, first_pixel))

    previous = start_pixel
    current = first_pixel

    while True:
        if current in pixel_to_node:
            return current, path, geodesic_length, "node"

        forward_neighbors = [nbr for nbr in neighbor_map[current] if nbr != previous]

        if len(forward_neighbors) == 0:
            return current, path, geodesic_length, "dead_end"

        if len(forward_neighbors) > 1:
            return current, path, geodesic_length, "ambiguous_branch"

        nxt = forward_neighbors[0]
        key = segment_key(current, nxt)
        if key in visited_segments:
            return current, path, geodesic_length, "revisited"

        visited_segments.add(key)
        geodesic_length += step_length(current, nxt)
        path.append(nxt)
        previous, current = current, nxt


def trace_cycle_component(
    component_pixels: Sequence[Pixel],
    neighbor_map: Dict[Pixel, List[Pixel]],
) -> Tuple[List[Pixel], float]:
    """Trace a pure cycle component that has no endpoints or junctions."""
    start = min(component_pixels)
    neighbors = neighbor_map[start]
    if len(neighbors) != 2:
        return [start], 0.0

    path = [start]
    geodesic_length = 0.0

    previous = start
    current = neighbors[0]
    path.append(current)
    geodesic_length += step_length(previous, current)

    while current != start:
        forward_neighbors = [nbr for nbr in neighbor_map[current] if nbr != previous]
        if not forward_neighbors:
            break
        nxt = forward_neighbors[0]
        geodesic_length += step_length(current, nxt)
        path.append(nxt)
        previous, current = current, nxt

    return path, geodesic_length


def add_node(
    graph: nx.MultiGraph,
    node_supports: Dict[int, set],
    pixel_to_node: Dict[Pixel, int],
    next_node_id: int,
    pixels: Sequence[Pixel],
    kind: str,
    component_id: int,
    reason: Optional[str] = None,
) -> int:
    """Create one graph node from one or more support pixels."""
    representative, centroid_row, centroid_col = choose_representative_pixel(pixels)
    graph.add_node(
        next_node_id,
        kind=kind,
        row=float(centroid_row),
        col=float(centroid_col),
        pixel_row=int(representative[0]),
        pixel_col=int(representative[1]),
        support_size=int(len(pixels)),
        component_id=int(component_id),
        reason=reason or "",
    )
    support = set(pixels)
    node_supports[next_node_id] = support
    for pixel in support:
        pixel_to_node[pixel] = next_node_id
    return next_node_id


def prune_spurs(graph: nx.MultiGraph, min_spur_length: float) -> int:
    """
    Iteratively remove very short terminal branches.

    Disabled by default because research workflows often prefer the raw graph.
    """
    if min_spur_length <= 0:
        return 0

    removed = 0
    changed = True

    while changed:
        changed = False
        for node in list(graph.nodes):
            node_kind = graph.nodes[node].get("kind")
            incident_edges = list(graph.edges(node, keys=True, data=True))
            if node_kind != "endpoint" or len(incident_edges) != 1:
                continue

            u, v, key, edge_data = incident_edges[0]
            if float(edge_data.get("geodesic_length", 0.0)) > min_spur_length:
                continue

            graph.remove_edge(u, v, key)
            removed += 1
            changed = True

            if node in graph and len(list(graph.edges(node, keys=True))) == 0:
                graph.remove_node(node)

            opposite = v if u == node else u
            if opposite in graph and len(list(graph.edges(opposite, keys=True))) == 0:
                graph.remove_node(opposite)
            break

    return removed


def build_graph_from_skeleton(
    skeleton: np.ndarray,
    source_path: Path,
    config: ExtractionConfig,
) -> nx.MultiGraph:
    """Convert one skeleton image into a compressed NetworkX MultiGraph."""
    original_pixels = int(skeleton.sum())
    cleaned_skeleton, removed_components, removed_pixels = prune_small_components(
        skeleton,
        min_component_pixels=config.min_component_pixels,
    )

    graph = nx.MultiGraph()
    graph.graph.update(
        {
            "graph_type": "finger_vein_skeleton_multigraph",
            "source_path": str(source_path),
            "image_height": int(cleaned_skeleton.shape[0]),
            "image_width": int(cleaned_skeleton.shape[1]),
            "original_skeleton_pixels": int(original_pixels),
            "skeleton_pixels": int(cleaned_skeleton.sum()),
            "min_component_pixels": int(config.min_component_pixels),
            "removed_small_components": int(removed_components),
            "removed_small_component_pixels": int(removed_pixels),
            "min_spur_length": float(config.min_spur_length),
        }
    )

    if not np.any(cleaned_skeleton):
        graph.graph.update(
            {
                "nodes": 0,
                "edges": 0,
                "connected_components": 0,
                "endpoints": 0,
                "junctions": 0,
                "isolated_nodes": 0,
                "cycle_nodes": 0,
                "synthetic_nodes": 0,
                "self_loops": 0,
                "total_geodesic_length": 0.0,
                "spur_edges_removed": 0,
            }
        )
        return graph

    neighbor_map = build_neighbor_map(cleaned_skeleton)
    degree_map = {pixel: len(neighbors) for pixel, neighbors in neighbor_map.items()}

    binary = cleaned_skeleton.astype(np.uint8)
    num_components, component_labels = cv2.connectedComponents(binary, connectivity=8)

    junction_mask = np.zeros_like(binary, dtype=np.uint8)
    for pixel, degree in degree_map.items():
        if degree >= 3:
            junction_mask[pixel] = 1

    junction_components, junction_labels = cv2.connectedComponents(junction_mask, connectivity=8)

    node_supports: Dict[int, set] = {}
    pixel_to_node: Dict[Pixel, int] = {}
    next_node_id = 0

    # Merge multi-pixel junction neighborhoods into one graph node.
    for label in range(1, junction_components):
        pixels = [tuple(map(int, rc)) for rc in np.argwhere(junction_labels == label)]
        if not pixels:
            continue
        component_id = int(component_labels[pixels[0]])
        add_node(
            graph=graph,
            node_supports=node_supports,
            pixel_to_node=pixel_to_node,
            next_node_id=next_node_id,
            pixels=pixels,
            kind="junction",
            component_id=component_id,
        )
        next_node_id += 1

    # Endpoints and isolated pixels become their own nodes.
    for pixel in sorted(neighbor_map):
        if pixel in pixel_to_node:
            continue
        degree = degree_map[pixel]
        if degree == 1:
            kind = "endpoint"
        elif degree == 0:
            kind = "isolated"
        else:
            continue

        component_id = int(component_labels[pixel])
        add_node(
            graph=graph,
            node_supports=node_supports,
            pixel_to_node=pixel_to_node,
            next_node_id=next_node_id,
            pixels=[pixel],
            kind=kind,
            component_id=component_id,
        )
        next_node_id += 1

    visited_segments = set()

    for node_id in sorted(node_supports):
        support_pixels = sorted(node_supports[node_id])
        external_neighbors: Dict[Pixel, Pixel] = {}
        for start_pixel in support_pixels:
            for neighbor in neighbor_map[start_pixel]:
                if neighbor not in node_supports[node_id]:
                    external_neighbors.setdefault(neighbor, start_pixel)

        for neighbor in sorted(external_neighbors):
            start_pixel = external_neighbors[neighbor]
            key = segment_key(start_pixel, neighbor)
            if key in visited_segments:
                continue

            terminal_pixel, path, geodesic_length, status = trace_branch(
                start_pixel=start_pixel,
                first_pixel=neighbor,
                pixel_to_node=pixel_to_node,
                neighbor_map=neighbor_map,
                visited_segments=visited_segments,
            )

            end_node = pixel_to_node.get(terminal_pixel)
            if end_node is None:
                component_id = int(component_labels[terminal_pixel])
                add_node(
                    graph=graph,
                    node_supports=node_supports,
                    pixel_to_node=pixel_to_node,
                    next_node_id=next_node_id,
                    pixels=[terminal_pixel],
                    kind="synthetic",
                    component_id=component_id,
                    reason=status,
                )
                end_node = next_node_id
                next_node_id += 1

            end_attrs = graph.nodes[end_node]
            start_attrs = graph.nodes[node_id]
            euclidean_length = math.hypot(
                end_attrs["row"] - start_attrs["row"],
                end_attrs["col"] - start_attrs["col"],
            )
            tortuosity = geodesic_length / euclidean_length if euclidean_length > 0 else 1.0
            edge_component = int(component_labels[start_pixel])

            edge_payload = {
                "component_id": edge_component,
                "pixel_length": int(max(len(path) - 1, 0)),
                "geodesic_length": float(round(geodesic_length, 6)),
                "euclidean_length": float(round(euclidean_length, 6)),
                "tortuosity": float(round(tortuosity, 6)),
                "start_pixel_row": int(path[0][0]),
                "start_pixel_col": int(path[0][1]),
                "end_pixel_row": int(path[-1][0]),
                "end_pixel_col": int(path[-1][1]),
                "branch_type": "loop" if node_id == end_node else "segment",
            }
            if config.store_edge_pixels:
                edge_payload["path_pixels"] = path

            graph.add_edge(node_id, end_node, **edge_payload)

    # Components with no endpoints / junctions are pure cycles.
    existing_component_ids = {
        int(attrs["component_id"])
        for _, attrs in graph.nodes(data=True)
    }
    for component_id in range(1, num_components):
        if component_id in existing_component_ids:
            continue
        component_pixels = [
            tuple(map(int, rc))
            for rc in np.argwhere(component_labels == component_id)
        ]
        if not component_pixels:
            continue

        cycle_node_id = add_node(
            graph=graph,
            node_supports=node_supports,
            pixel_to_node=pixel_to_node,
            next_node_id=next_node_id,
            pixels=component_pixels,
            kind="cycle",
            component_id=component_id,
        )
        next_node_id += 1

        cycle_path, geodesic_length = trace_cycle_component(component_pixels, neighbor_map)
        edge_payload = {
            "component_id": int(component_id),
            "pixel_length": int(max(len(cycle_path) - 1, 0)),
            "geodesic_length": float(round(geodesic_length, 6)),
            "euclidean_length": 0.0,
            "tortuosity": 1.0,
            "start_pixel_row": int(cycle_path[0][0]),
            "start_pixel_col": int(cycle_path[0][1]),
            "end_pixel_row": int(cycle_path[-1][0]),
            "end_pixel_col": int(cycle_path[-1][1]),
            "branch_type": "loop",
        }
        if config.store_edge_pixels:
            edge_payload["path_pixels"] = cycle_path
        graph.add_edge(cycle_node_id, cycle_node_id, **edge_payload)

    spur_edges_removed = prune_spurs(graph, config.min_spur_length)

    total_geodesic_length = sum(
        float(data.get("geodesic_length", 0.0))
        for _, _, data in graph.edges(data=True)
    )

    for node_id in list(graph.nodes):
        incident_edges = list(graph.edges(node_id, keys=True))
        graph.nodes[node_id]["graph_degree"] = int(len(incident_edges))

    graph.graph.update(
        {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "connected_components": int(nx.number_connected_components(graph)) if graph.number_of_nodes() else 0,
            "endpoints": int(sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("kind") == "endpoint")),
            "junctions": int(sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("kind") == "junction")),
            "isolated_nodes": int(sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("kind") == "isolated")),
            "cycle_nodes": int(sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("kind") == "cycle")),
            "synthetic_nodes": int(sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("kind") == "synthetic")),
            "self_loops": int(nx.number_of_selfloops(graph)),
            "total_geodesic_length": float(round(total_geodesic_length, 6)),
            "spur_edges_removed": int(spur_edges_removed),
        }
    )

    return graph


def save_graph(graph: nx.MultiGraph, out_path: Path) -> None:
    """Persist a NetworkX graph with standard pickle serialization."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: Path) -> nx.MultiGraph:
    """Load one previously saved graph pickle."""
    with path.open("rb") as f:
        return pickle.load(f)


def summarise_graph(graph: nx.MultiGraph, graph_path: Path) -> Dict[str, object]:
    """Create a flat record suitable for CSV logging and quick inspection."""
    source_path = Path(graph.graph.get("source_path", ""))
    parts = source_path.parts
    session = parts[-3] if len(parts) >= 3 else ""
    subject_finger = parts[-2] if len(parts) >= 2 else ""
    image_name = parts[-1] if len(parts) >= 1 else ""

    return {
        "graph_path": str(graph_path),
        "source_path": str(source_path),
        "session": session,
        "subject_finger": subject_finger,
        "image_name": image_name,
        "image_height": int(graph.graph.get("image_height", 0)),
        "image_width": int(graph.graph.get("image_width", 0)),
        "original_skeleton_pixels": int(graph.graph.get("original_skeleton_pixels", 0)),
        "skeleton_pixels": int(graph.graph.get("skeleton_pixels", 0)),
        "removed_small_components": int(graph.graph.get("removed_small_components", 0)),
        "removed_small_component_pixels": int(graph.graph.get("removed_small_component_pixels", 0)),
        "nodes": int(graph.graph.get("nodes", graph.number_of_nodes())),
        "edges": int(graph.graph.get("edges", graph.number_of_edges())),
        "connected_components": int(graph.graph.get("connected_components", 0)),
        "endpoints": int(graph.graph.get("endpoints", 0)),
        "junctions": int(graph.graph.get("junctions", 0)),
        "isolated_nodes": int(graph.graph.get("isolated_nodes", 0)),
        "cycle_nodes": int(graph.graph.get("cycle_nodes", 0)),
        "synthetic_nodes": int(graph.graph.get("synthetic_nodes", 0)),
        "self_loops": int(graph.graph.get("self_loops", 0)),
        "total_geodesic_length": float(graph.graph.get("total_geodesic_length", 0.0)),
        "spur_edges_removed": int(graph.graph.get("spur_edges_removed", 0)),
    }


def write_summary_csv(records: Sequence[Dict[str, object]], csv_path: Path) -> None:
    """Write the dataset-level graph summary CSV."""
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "graph_path",
        "source_path",
        "session",
        "subject_finger",
        "image_name",
        "image_height",
        "image_width",
        "original_skeleton_pixels",
        "skeleton_pixels",
        "removed_small_components",
        "removed_small_component_pixels",
        "nodes",
        "edges",
        "connected_components",
        "endpoints",
        "junctions",
        "isolated_nodes",
        "cycle_nodes",
        "synthetic_nodes",
        "self_loops",
        "total_geodesic_length",
        "spur_edges_removed",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_batch(
    skeleton_root: Path = SKELETON_ROOT,
    graph_root: Path = GRAPH_ROOT,
    summary_csv: Path = SUMMARY_CSV,
    log_file: Path = LOG_FILE,
    skip_existing: bool = SKIP_EXISTING,
    min_component_pixels: int = MIN_COMPONENT_PIXELS,
    min_spur_length: float = MIN_SPUR_LENGTH,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Batch-build graphs for every skeleton image in the results tree."""
    config = ExtractionConfig(
        skeleton_root=skeleton_root,
        graph_root=graph_root,
        summary_csv=summary_csv,
        log_file=log_file,
        skip_existing=skip_existing,
        min_component_pixels=min_component_pixels,
        min_spur_length=min_spur_length,
    )

    pattern = str(skeleton_root / "**" / "*_binary.png")
    skeleton_files = sorted(Path(p) for p in glob.glob(pattern, recursive=True))

    if limit is not None:
        skeleton_files = skeleton_files[:limit]

    if not skeleton_files:
        print(f"No *_binary.png files found under '{skeleton_root}'.")
        print("Run 05_skeletize.py first.")
        return []

    total = len(skeleton_files)

    print("=" * 60)
    print("  TopoVein — File 6: Graph Extraction")
    print("=" * 60)
    print(f"  Skeleton images   : {total}")
    print(f"  Output root       : {graph_root}/")
    print(f"  Min component px  : {min_component_pixels}")
    print(f"  Spur prune length : {min_spur_length}")
    print()

    processed = 0
    skipped = 0
    failed: List[Tuple[str, str]] = []
    records: List[Dict[str, object]] = []
    log_lines: List[str] = []
    t_start = time.time()

    for index, skeleton_path in enumerate(skeleton_files, 1):
        out_path = compute_output_path(skeleton_path, skeleton_root, graph_root)

        if index % 20 == 0 or index == 1 or index == total:
            elapsed = time.time() - t_start
            eta = (elapsed / index) * (total - index) if index > 1 else 0.0
            print(f"  [{index:>4}/{total}]  ETA {eta:.0f}s  →  {skeleton_path.name}")

        try:
            if skip_existing and out_path.exists():
                graph = load_graph(out_path)
                records.append(summarise_graph(graph, out_path))
                skipped += 1
                log_lines.append(f"SKIP {skeleton_path} -> {out_path}")
                continue

            skeleton = load_skeleton(skeleton_path)
            if skeleton is None:
                raise RuntimeError("cv2.imread failed")

            graph = build_graph_from_skeleton(skeleton, skeleton_path, config)
            save_graph(graph, out_path)
            records.append(summarise_graph(graph, out_path))
            processed += 1

            log_lines.append(
                "OK  "
                f"nodes={graph.graph.get('nodes', 0):>4}  "
                f"edges={graph.graph.get('edges', 0):>4}  "
                f"loops={graph.graph.get('self_loops', 0):>3}  "
                f"len={graph.graph.get('total_geodesic_length', 0.0):>8.2f}  "
                f"{out_path}"
            )
        except Exception as exc:  # noqa: BLE001 - batch pipeline should keep going
            failed.append((str(skeleton_path), str(exc)))
            log_lines.append(f"FAIL {skeleton_path} :: {exc}")

    write_summary_csv(records, summary_csv)

    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print("  GRAPH EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Processed        : {processed}")
    print(f"  Skipped          : {skipped}")
    print(f"  Failed           : {len(failed)}")
    print(f"  Time elapsed     : {elapsed:.1f}s")

    if records:
        node_counts = [int(record["nodes"]) for record in records]
        edge_counts = [int(record["edges"]) for record in records]
        lengths = [float(record["total_geodesic_length"]) for record in records]
        print()
        print("  AGGREGATE GRAPH STATS")
        print(f"    Nodes/image    — mean={np.mean(node_counts):.1f}  min={min(node_counts)}  max={max(node_counts)}")
        print(f"    Edges/image    — mean={np.mean(edge_counts):.1f}  min={min(edge_counts)}  max={max(edge_counts)}")
        print(f"    Length/image   — mean={np.mean(lengths):.1f}  min={min(lengths):.1f}  max={max(lengths):.1f}")
        print(f"\n  Summary CSV saved → {summary_csv}")

    if failed:
        print()
        print("  FAIL EXAMPLES (first 10):")
        for path, err in failed[:10]:
            print(f"    ✗ {Path(path).name}  ::  {err}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"  Log saved        → {log_file}")

    return records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch graph extraction."""
    parser = argparse.ArgumentParser(description="Convert skeleton images into NetworkX graphs.")
    parser.add_argument(
        "--skeleton-root",
        type=Path,
        default=SKELETON_ROOT,
        help="Root directory containing skeleton images.",
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=GRAPH_ROOT,
        help="Output directory for graph pickle files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=SUMMARY_CSV,
        help="Where to save the dataset-level graph summary CSV.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=LOG_FILE,
        help="Where to save the batch log.",
    )
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=MIN_COMPONENT_PIXELS,
        help="Remove disconnected skeleton fragments smaller than this size.",
    )
    parser.add_argument(
        "--min-spur-length",
        type=float,
        default=MIN_SPUR_LENGTH,
        help="Optionally prune very short terminal branches after graph extraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N skeleton images (useful for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild graph files even if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch(
        skeleton_root=args.skeleton_root,
        graph_root=args.graph_root,
        summary_csv=args.summary_csv,
        log_file=args.log_file,
        skip_existing=not args.overwrite,
        min_component_pixels=args.min_component_pixels,
        min_spur_length=args.min_spur_length,
        limit=args.limit,
    )
