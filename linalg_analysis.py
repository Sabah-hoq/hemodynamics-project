"""
Linear-algebra diagnostics on the vessel network's nodal system A V = b.

solve_network() in solver.py already does the core linear algebra of the
project: it assembles a conductance (weighted-graph-Laplacian) matrix from
Kirchhoff's current law and solves A V = b for the unknown node pressures.
This module adds the analysis layer on top of that system:

  - the condition number of A, which says how much solver round-off / a
    tiny change in resistance can be amplified in the computed pressures
  - a sensitivity matrix dP_leaf/dR_vessel obtained by perturbing the
    linear system, which says which sensor is most informative about
    which vessel -- a model-free, purely linear-algebra counterpart to
    the Random Forest's impurity-based feature importance
  - the weighted graph Laplacian of the whole tree (including the ground
    node) and its spectrum; the second-smallest eigenvalue (algebraic
    connectivity / Fiedler value) measures how "well-connected" the flow
    network is as a graph, independent of any particular pressure solve
"""

import numpy as np

from solver import build_system, solve_network
from constants import P_IN, P_OUT


def condition_number(net, p_in=P_IN, p_out=P_OUT):
    """np.linalg.cond(A): the ratio of A's largest to smallest singular
    value. A well-posed solve has this near 1; a network with enormous
    terminal resistances next to small internal ones can push this high,
    meaning pressure estimates become sensitive to numerical round-off."""
    A, b, index, unknown_nodes, fixed = build_system(net, p_in, p_out)
    cond = np.linalg.cond(A)
    eigvals = np.linalg.eigvalsh((A + A.T) / 2)  # A is already symmetric; guard float noise
    return {
        "condition_number": float(cond),
        "A_shape": A.shape,
        "eigenvalues": eigvals,
        "min_eigenvalue": float(eigvals.min()),
        "max_eigenvalue": float(eigvals.max()),
    }


def sensitivity_matrix(net, p_in=P_IN, p_out=P_OUT, rel_step=1e-3):
    """Finite-difference Jacobian d(leaf pressure)/d(log resistance) for
    every named vessel, obtained by perturbing one vessel's resistance at
    a time, re-solving A V = b, and measuring the change at every leaf.
    Using d/d(log R) (rather than d/dR) puts every vessel on a comparable
    scale despite resistance spanning many orders of magnitude across
    generations. Returns a (n_leaves x n_named_vessels) array plus the
    row/column labels.
    """
    net.reset()
    base_pressures, _ = solve_network(net, p_in, p_out)
    leaves = net.leaves

    named_vessels = [v for v in net.vessels if v.v != net.GROUND]
    J = np.zeros((len(leaves), len(named_vessels)))

    for j, ves in enumerate(named_vessels):
        original_R = ves.resistance
        step = original_R * rel_step
        d_logR = np.log(original_R + step) - np.log(original_R)

        ves.resistance = original_R + step
        perturbed_pressures, _ = solve_network(net, p_in, p_out)
        ves.resistance = original_R  # restore before the next vessel

        for i, leaf in enumerate(leaves):
            dP = perturbed_pressures[leaf] - base_pressures[leaf]
            J[i, j] = dP / d_logR

    net.reset()
    return {
        "J": J,
        "leaf_labels": [f"node{n}" for n in leaves],
        "vessel_labels": [f"v{ves.id}" for ves in named_vessels],
        "vessel_ids": [ves.id for ves in named_vessels],
    }


def graph_laplacian(net):
    """Weighted graph Laplacian L = D - W of the whole network (internal
    nodes + root + ground), with edge weights = conductance 1/R. This is
    the same conductance information as A in build_system, but as a full
    symmetric graph operator rather than one restricted to unknown nodes
    -- the natural object for spectral graph analysis.
    """
    nodes = sorted(net.nodes)  # includes 0 and net.GROUND
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))

    for ves in net.vessels:
        g = 1.0 / ves.resistance
        iu, iv = idx[ves.u], idx[ves.v]
        W[iu, iv] += g
        W[iv, iu] += g
        if ves.leak_conductance > 0:
            # a leak bleeds conductance from this vessel's downstream node
            # straight to ground, regardless of where ground sits in `nodes`
            ig = idx[net.GROUND]
            W[iv, ig] += ves.leak_conductance
            W[ig, iv] += ves.leak_conductance

    D = np.diag(W.sum(axis=1))
    L = D - W
    return L, nodes


def algebraic_connectivity(net):
    """Eigenvalues of the graph Laplacian, sorted ascending. The smallest
    is always ~0 (the network is one connected component); the second
    smallest -- the Fiedler value -- measures overall connectivity: it
    drops toward 0 as the network is pinched down to a near-bottleneck
    (e.g. a severe clog on the single vessel feeding a whole subtree).
    """
    L, nodes = graph_laplacian(net)
    eigvals = np.linalg.eigvalsh((L + L.T) / 2)
    eigvals = np.sort(eigvals)
    return {
        "eigenvalues": eigvals,
        "fiedler_value": float(eigvals[1]) if len(eigvals) > 1 else float("nan"),
        "nodes": nodes,
    }
