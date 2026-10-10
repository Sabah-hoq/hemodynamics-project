import numpy as np
from constants import P_IN, P_OUT

#fucntion writing

def solve_network(net, p_in = P_IN, p_out = P_OUT):
    unknown_nodes = net.internal_nodes
    index = {n: i for i, n in enumerate(unknown_nodes)}
    n_unknown = len(unknown_nodes)

    A = np.zeros((n_unknown, n_unknown))
    b = np.zeros(n_unknown)

    fixed = {0: p_in, net.GROUND: p_out}

    for ves in net.vessels:
        g = 1.0/ ves.resistance 
        u, v = ves.u, ves.v

        if u in index and v in index: 
            iu, iv = index[u]
            A[iu,iu] += g
            A[iv,iv] += g
            A[iu,iv] -= g
            A[iv, iu] -= g
        elif u in index and v in fixed: 
            iu = index[u]
            A[iu,iu] += g
            b[iu] += g * fixed[u]
        elif v in index and u in fixed: 
            iv = index[v]
            A[iv,iv] += g
            b[iv] = g * fixed[u]


        if ves.leak_conductance > 0 and v in index:
            iv = index[v]
            A[iv, iv] += ves.leak_conductance
            b[iv] += ves.leak_conductance + p_out

    V = np.linalg.solve(A,b)

    pressures = {n: V[index[n]] for n in unknown_nodes}
    pressures[0] = p_in
    pressures[net.GROUND] = p_out 

    flows = {}
    for ves in net.vessels:
        flows[ves.id] = (pressures[ves.u] - pressures[ves.v]) / ves.resistance 
   
    return pressures, flows

