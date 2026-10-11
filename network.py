import numpy as np 

from constants import BLOOD_VISCOSITY

MURRAY_RATIO = 2 ** (-1/3)
LENGTH_RATIO = 0.8 

def poiseuille_resistance(radius, length, viscosity=BLOOD_VISCOSITY):
     """Hagen-Poiseuille hydraulic resistance: R = 8*eta*L / (pi*r^4)."""
     return 8.0 * viscosity * length / (np.pi * radius ** 4)

class Vessel:
     def __init__(self, vessel_id, u, v, radius=None, length=None, 
                  explicit_resistance=None):
          self.id = vessel_id
          self.u = u
          self.v = v
          self.radius = radius 
          self.legnth = length 

          if explicit_resistance is not None: 
               self.base_resistance = explicit_resistance
          else: 
              self.base_resistance = poiseuille_resistance(radius, length)

          self.resistance = self.base_resistance
          self.leak_conductance = 0.0

     def reset(self):
          self.resistance = self.base_resistance
          self.leak_conductance = 0.0

     def __repr__(self):
          return f"Vessel(id={self.id}, {self.u}->{self.v},  = {self.resistance:.3e})"

class VesselNetwork:
    GROUND = -1

    def __init__(self, generations=4, root_radius=4e-3, root_length=0.03,
                 terminal_resistance=3e8):
        self.generations = generations
        self.root_radius = root_radius
        self.root_length = root_length
        self.terminal_resistance = terminal_resistance

        self.vessels = []
        self.nodes = {0}
        self._build()

    def _build(self):
        vid = 0
        radius = self.root_radius
        length = self.root_length

        frontier = [0]
        for _ in range(self.generations):
            next_frontier = []
            child_radius = radius * MURRAY_RATIO
            child_length = length * LENGTH_RATIO
            for parent in frontier:
                for offset in (1, 2):
                    child = 2 * parent + offset
                    self.vessels.append(Vessel(vid, parent, child, child_radius, child_length))
                    vid += 1
                    self.nodes.add(child)
                    next_frontier.append(child)
            frontier = next_frontier
            radius, length = child_radius, child_length

        self.leaves = frontier

        for leaf in self.leaves:
            self.vessels.append(
                Vessel(vid, leaf, self.GROUND, explicit_resistance=self.terminal_resistance)
            )
            vid += 1

        self.nodes.add(self.GROUND)
        self.internal_nodes = sorted(n for n in self.nodes if n not in (0, self.GROUND))
        self.n_named_vessels = len(self.vessels) - len(self.leaves)  # excludes terminal edges

    def reset(self):
        for ves in self.vessels:
            ves.reset()
