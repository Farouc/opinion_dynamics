"""Graph generation utilities for zealot voter simulations."""

from __future__ import annotations

import networkx as nx


def _to_integer_labels(G: nx.Graph) -> nx.Graph:
    """Return an integer-labeled copy to keep state arrays aligned with node ids."""
    if set(G.nodes()) == set(range(G.number_of_nodes())):
        return G
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def generate_erdos_renyi(n: int, p: float, seed: int | None = None) -> nx.Graph:
    """Generate an Erdos-Renyi graph G(n, p)."""
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    return _to_integer_labels(G)


def generate_fully_connected(n: int) -> nx.Graph:
    """Generate a fully connected (complete) graph on n nodes."""
    G = nx.complete_graph(n=n)
    return _to_integer_labels(G)


def generate_barabasi_albert(n: int, m: int, seed: int | None = None) -> nx.Graph:
    """Generate a Barabasi-Albert preferential attachment graph."""
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    return _to_integer_labels(G)


def generate_grid_lattice(L: int) -> nx.Graph:
    """Generate a 2D L x L grid lattice graph with integer node labels."""
    G = nx.grid_2d_graph(L, L)
    return _to_integer_labels(G)


def create_graph_from_config(config: dict, seed: int | None = None) -> nx.Graph:
    """Create a graph from configuration entries.

    Expected keys:
    - graph_type: one of {"fully_connected", "erdos_renyi", "barabasi_albert", "grid_lattice"}
    - n, p, m, L depending on graph type
    """
    graph_type = str(config.get("graph_type", "erdos_renyi")).lower()

    if graph_type == "fully_connected":
        return generate_fully_connected(n=int(config["n"]))

    if graph_type == "erdos_renyi":
        return generate_erdos_renyi(
            n=int(config["n"]),
            p=float(config["p"]),
            seed=seed,
        )

    if graph_type == "barabasi_albert":
        return generate_barabasi_albert(
            n=int(config["n"]),
            m=int(config.get("m", 3)),
            seed=seed,
        )

    if graph_type == "grid_lattice":
        return generate_grid_lattice(L=int(config["L"]))

    raise ValueError(
        f"Unsupported graph_type='{graph_type}'. "
        "Choose from {'fully_connected', 'erdos_renyi', 'barabasi_albert', 'grid_lattice'}."
    )
