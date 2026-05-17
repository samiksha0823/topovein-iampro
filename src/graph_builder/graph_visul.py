import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
# Locate graph directory automatically

root = Path.cwd()

target_subpath = Path("results") / "graphs" / "1st_session" / "001_1"

possible_paths = [root / target_subpath]
possible_paths += [parent / target_subpath for parent in root.parents]

data_dir = next((p for p in possible_paths if p.exists()), None)

if data_dir is None:
    raise FileNotFoundError(
        f"Graph directory not found.\nChecked:\n"
        + "\n".join(str(p) for p in possible_paths)
    )

print(f"[INFO] Graph directory found: {data_dir}")


graph_files = sorted(data_dir.glob("*_graph.pkl"))

if len(graph_files) == 0:
    raise FileNotFoundError("No graph pickle files found.")

print(f"[INFO] Found {len(graph_files)} graph files")


def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{graph_path.name} is not a NetworkX graph.")

    return graph


def analyze_graph(graph, name="Graph"):

    print("\n" + "=" * 60)
    print(f"{name}")
    print("=" * 60)

    print(f"Nodes                : {graph.number_of_nodes()}")
    print(f"Edges                : {graph.number_of_edges()}")

    degrees = [d for _, d in graph.degree()]

    print(f"Average Degree       : {np.mean(degrees):.2f}")
    print(f"Max Degree           : {np.max(degrees)}")

    components = nx.number_connected_components(graph)
    print(f"Connected Components : {components}")

    density = nx.density(graph)
    print(f"Graph Density        : {density:.6f}")


def visualize_graph(graph, title="Graph Visualization"):

    plt.figure(figsize=(12, 10))

    # spring layout is better for topology visualization
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=15,
        alpha=0.8
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        width=0.5,
        alpha=0.5
    )

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


all_stats = []

for graph_file in graph_files:

    print(f"\n[INFO] Loading {graph_file.name}")

    graph = load_graph(graph_file)

    analyze_graph(graph, graph_file.name)

    visualize_graph(graph, graph_file.name)

    all_stats.append({
        "graph_name": graph_file.name,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "components": nx.number_connected_components(graph),
    })


stats_df = pd.DataFrame(all_stats)

print("\nGraph Summary:")
nx.display(stats_df)