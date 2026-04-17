"""Lightweight Weisfeiler-Lehman (WL) structural features."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import networkx as nx
import numpy as np


def wl_color_refinement(
    G: nx.Graph,
    n_iter: int = 3,
    initial_labels: dict[int, int] | None = None,
) -> dict:
    """Run 1-WL color refinement and return node class labels.

    Parameters
    ----------
    G : nx.Graph
    n_iter : int
        Maximum number of refinement iterations.
    initial_labels : dict[int, int] | None
        Optional initial labels. Defaults to node degree labels.
    """
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative.")

    if initial_labels is None:
        labels = {int(node): int(G.degree(node)) for node in G.nodes()}
    else:
        labels = {int(node): int(lbl) for node, lbl in initial_labels.items()}

    history: list[dict[int, int]] = [labels.copy()]

    for _ in range(int(n_iter)):
        signatures: dict[int, tuple[int, tuple[int, ...]]] = {}
        for node in G.nodes():
            node_i = int(node)
            neigh_labels = sorted(int(labels[int(nb)]) for nb in G.neighbors(node_i))
            signatures[node_i] = (int(labels[node_i]), tuple(neigh_labels))

        unique = sorted(set(signatures.values()), key=lambda x: (x[0], len(x[1]), x[1]))
        remap = {sig: idx for idx, sig in enumerate(unique)}

        new_labels = {node: int(remap[sig]) for node, sig in signatures.items()}
        history.append(new_labels.copy())

        if new_labels == labels:
            labels = new_labels
            break

        labels = new_labels

    class_counts = Counter(labels.values())
    return {
        "labels": labels,
        "history": history,
        "n_classes": int(len(class_counts)),
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
    }


def _entropy_from_counts(counts: np.ndarray) -> tuple[float, float]:
    if counts.size == 0:
        return 0.0, 0.0
    probs = counts / np.sum(counts)
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-15, 1.0))))
    if counts.size <= 1:
        return entropy, 0.0
    return entropy, float(entropy / np.log(counts.size))


def _topk_hist_features(counts: np.ndarray, topk: int = 5, prefix: str = "wl_hist") -> dict[str, float]:
    out: dict[str, float] = {}
    if counts.size == 0:
        for i in range(topk):
            out[f"{prefix}_top{i + 1}_fraction"] = 0.0
        out[f"{prefix}_n_singletons"] = 0.0
        return out

    counts_sorted = np.sort(counts)[::-1]
    probs = counts_sorted / np.sum(counts_sorted)
    for i in range(topk):
        out[f"{prefix}_top{i + 1}_fraction"] = float(probs[i]) if i < probs.size else 0.0
    out[f"{prefix}_n_singletons"] = float(np.sum(counts == 1))
    return out


def compute_wl_coverage_features(
    G: nx.Graph,
    zealot_nodes: Iterable[int],
    n_iter: int = 3,
    wl_labels: dict[int, int] | None = None,
) -> dict[str, float | dict]:
    """Compute WL class coverage features for one zealot set."""
    zealot_list = [int(node) for node in zealot_nodes]
    zealot_set = set(zealot_list)

    if wl_labels is None:
        wl_labels = wl_color_refinement(G, n_iter=n_iter)["labels"]
    wl_labels = {int(node): int(lbl) for node, lbl in wl_labels.items()}

    all_classes = [int(lbl) for lbl in wl_labels.values()]
    total_classes = len(set(all_classes))
    class_sizes = Counter(all_classes)

    zealot_classes = [int(wl_labels[node]) for node in zealot_set] if zealot_set else []
    zealot_counts = Counter(zealot_classes)
    covered_classes = set(zealot_counts.keys())

    counts_arr = np.asarray(list(zealot_counts.values()), dtype=float)
    entropy, entropy_norm = _entropy_from_counts(counts_arr)

    avg_class_size = float(
        np.mean([class_sizes[c] for c in zealot_classes]) if zealot_classes else 0.0
    )

    out: dict[str, float | dict] = {
        "wl_total_classes": float(total_classes),
        "wl_covered_classes": float(len(covered_classes)),
        "wl_covered_fraction": float(len(covered_classes) / total_classes) if total_classes > 0 else 0.0,
        "wl_entropy": float(entropy),
        "wl_entropy_normalized": float(entropy_norm),
        "wl_avg_class_size_for_zealots": float(avg_class_size),
        "wl_histogram": {str(k): int(v) for k, v in sorted(zealot_counts.items())},
    }
    out.update(_topk_hist_features(counts_arr, topk=5, prefix="wl_hist"))
    return out


def compute_two_camp_wl_features(
    G: nx.Graph,
    pos_nodes: Iterable[int],
    neg_nodes: Iterable[int],
    n_iter: int = 3,
    wl_labels: dict[int, int] | None = None,
) -> dict[str, float | dict]:
    """Compute WL features for two camps and their overlap."""
    if wl_labels is None:
        wl_labels = wl_color_refinement(G, n_iter=n_iter)["labels"]
    wl_labels = {int(node): int(lbl) for node, lbl in wl_labels.items()}

    pos = compute_wl_coverage_features(G, pos_nodes, n_iter=n_iter, wl_labels=wl_labels)
    neg = compute_wl_coverage_features(G, neg_nodes, n_iter=n_iter, wl_labels=wl_labels)

    pos_set = {int(wl_labels[int(node)]) for node in pos_nodes}
    neg_set = {int(wl_labels[int(node)]) for node in neg_nodes}

    inter = pos_set.intersection(neg_set)
    union = pos_set.union(neg_set)

    out: dict[str, float | dict] = {
        **{f"pos_{k}": v for k, v in pos.items()},
        **{f"neg_{k}": v for k, v in neg.items()},
        "wl_overlap_classes": float(len(inter)),
        "wl_overlap_fraction_union": float(len(inter) / len(union)) if len(union) > 0 else 0.0,
        "wl_overlap_fraction_pos": float(len(inter) / len(pos_set)) if len(pos_set) > 0 else 0.0,
        "wl_overlap_fraction_neg": float(len(inter) / len(neg_set)) if len(neg_set) > 0 else 0.0,
    }

    return out
