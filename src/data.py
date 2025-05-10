import json
import logging
import random

import matplotlib.pyplot as plt
import networkx as nx

from src.utils.oracle_llms import ASK_ORACLE_MODEL

logger = logging.getLogger(__name__)


def draw_graph_with_minimal_edge_overlap(
    G, node_labels=None, node_size=1500, edge_color="black", node_color="lightblue"
):
    """
    Draw a graph with minimal edge overlap and custom node labels.

    Parameters:
    - G: NetworkX graph object
    - node_labels: Dictionary mapping node IDs to custom labels (if None, node IDs are used)
    - node_size: Size of nodes in the plot
    - edge_color: Color of edges
    - node_color: Color of nodes

    Returns:
    - pos: The node positions used for drawing
    """
    # If no custom labels provided, use node IDs as labels
    if node_labels is None:
        node_labels = {node: str(node) for node in G.nodes()}

    # Create figure with adequate size
    plt.figure(figsize=(12, 10))

    # Use Kamada-Kawai layout which typically gives good results with minimal edge crossings
    pos = nx.kamada_kawai_layout(G)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)

    # Draw node labels with custom names
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, font_size=10, font_weight="bold"
    )

    # Draw edges with curved paths to reduce visual overlap
    curved_edges(G, pos, edge_color=edge_color)

    # If the graph has edge labels, draw them
    if any("label" in G[u][v] for u, v in G.edges()):
        edge_labels = {(u, v): G[u][v].get("label", "") for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Connections")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def curved_edges(G, pos, curve_factor=0.1, edge_color="black", alpha=0.3):
    """Draw curved edges to reduce visual overlap."""
    ax = plt.gca()

    # Draw each edge with appropriate curvature
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Skip self-loops for simplicity
        if u == v:
            continue

        # Draw the curved edge
        ax.add_patch(
            plt.matplotlib.patches.FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle=f"arc3,rad={curve_factor}",
                arrowstyle="-",
                color=edge_color,
                alpha=alpha,
                linewidth=1.5,
            )
        )


def verify_connectivity(G):
    """Check if all nodes have at least one connection"""
    return len(list(nx.isolates(G))) == 0


def check_label_uniqueness_per_node(G):
    """Check if any node has duplicate edge labels to different neighbors"""
    has_duplicate_labels = False
    for node in G.nodes():
        labels_used = [G[node][neighbor]["label"] for neighbor in G.neighbors(node)]
        if len(labels_used) != len(set(labels_used)):
            print(f"Node {node} has duplicate edge labels: {labels_used}")
            has_duplicate_labels = True
            break
    return has_duplicate_labels is False


def create_single_connection_graph(
    n: int, edge_per_node_limit: int, possible_edge_labels: list[str]
) -> nx.Graph:
    """
    Create a random weighted graph with string labels as edges.

    Parameters:
    - n: Number of nodes
    - p: Probability of edge creation between any two nodes
    - possible_edge_labels: List of possible string labels for edges

    Properties:
    1. All nodes are connected to at least one other node
    2. No node connects to two different nodes with the same edge label

    Returns:
    - G: NetworkX graph object
    """

    def maybe_add_edge(u, v):
        # Check if the edge already exists
        if not G.has_edge(u, v):
            # Get labels already used by node u and node v
            u_used_labels = [G[u][neighbor]["label"] for neighbor in G.neighbors(u)]
            v_used_labels = [G[v][neighbor]["label"] for neighbor in G.neighbors(v)]

            # Find available labels
            available_labels = [
                label
                for label in possible_edge_labels
                if label not in u_used_labels and label not in v_used_labels
            ]

            # Add edge if there's an available label
            if available_labels:
                label = random.choice(available_labels)
                # Store the label as 'label' attribute instead of 'weight'
                G.add_edge(u, v, label=label)

                # logger.debug(f"Adding edge {u} --- {v} with label {label}")
                return True
        return False

    G = nx.Graph()
    G.add_nodes_from(range(n))

    nodes = list(range(n))
    edge_counter = {n: 0 for n in nodes}

    # Create a random graph
    for u in range(n):
        candidate_vs = [v for v in range(n) if v != u]
        random.shuffle(candidate_vs)
        for v in candidate_vs:
            # Check if the edge already exists
            if G.has_edge(u, v):
                continue

            # Check if adding this edge would exceed the limit
            if (
                edge_counter[u] < edge_per_node_limit
                and edge_counter[v] < edge_per_node_limit
            ):
                # Add the edge
                if maybe_add_edge(u, v):
                    edge_counter[u] += 1
                    edge_counter[v] += 1

    tests = [verify_connectivity, check_label_uniqueness_per_node]

    for test in tests:
        result = test(G)
        label = "PASS" if result else "FAIL"
        logger.info(f"{label} : {test.__name__}")

    return G
