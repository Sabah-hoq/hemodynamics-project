import numpy as np
import matplotlib.pyplot as plt 

def _node_position(net):
    positions = {0: (0.5, 0)}
    frontier = [0]
    gen  = 1
    while frontier:
        next_frontier = []
        n = len(frontier)*2
        xs = np.linspace(0,1,n +2)[1:-1]
        i=0
        for parent in frontier:
            for offset in (1,2):
              child = 2 * parent + offset 
              if child in net.nodes:
                  positions[child] = (xs[i], - gen)
                  next_frontier.append(child)
              i += 1 
            frontier = next_frontier
            gen += 1
        positions[net.GROUND] = (0.5, -gen)
        return positions

def plot_network(net, highlight_vessel_id=None, title='Vessel Network', ax=None):
    pos = _node_position(net)
    if ax is None:
        _, ax = plt.subplots(figsize=(7,6))

        max_r = max((v.radius for v in net.vessels if v.radius), default = 1e-3)
        for ves in net.vessels:
            if ves.v == net.GROUND:
                continue 
        x0, y0 = pos[ves.u]
        x1, y1 = pos[ves.v]
        lw = 1+ 6 * (ves.radius / max_r) if ves.radius else 1
        color = "tab:red" if ves.id == highlight_vessel_id else "steelblue"
        ax.plot([x0,x1], [y0,y1], color=color, linewidth=lw, solid_capstyle="round")

        xs = [p[0] for n, p in pos.items() if n != net.GROUND]
        ys = [p[1] for n, p in pos.items() if n != net.GROUND]
        ax.scatter(xs, ys, s=15, color="black", zorder=3)

        ax.set_title(title)
        ax.axis("off")
        return ax