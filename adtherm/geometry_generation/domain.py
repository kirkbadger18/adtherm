import numpy as np


class RigidCoordDomain:

    def __init__(self, indices):
        self.N_atoms = len(indices)
        self._assign_domain()

    def _assign_domain(self):
        self.vec1_frac: list = [0, 1]
        self.vec2_frac: list = [0, 1]
        self.dz: list = [-0.5, 1]
        self.alpha, self.beta, self.gamma = None, None, None
        self.N_dimensions = 3
        if self.N_atoms > 1:
            self.alpha: list = [-np.pi, np.pi]
            self.gamma: list = [-np.pi, np.pi]
            self.N_dimensions = 5
            if self.N_atoms > 2:
                self.beta: list = [-np.pi/2, np.pi/2]
                self.N_dimensions = 6

    def to_array(self) -> np.ndarray:
        if self.N_atoms == 1:
            return np.array([self.vec1_frac, self.vec2_frac, self.dz])
        elif self.N_atoms == 2:
            return np.array([self.vec1_frac, self.vec2_frac, self.dz,
                             self.alpha, self.gamma])
        else:
            return np.array([self.vec1_frac, self.vec2_frac, self.dz,
                             self.alpha, self.beta, self.gamma])

    def lower_bounds(self):
        array = self.to_array()
        lower_bounds = array[:, 0]
        return lower_bounds

    def upper_bounds(self):
        array = self.to_array()
        upper_bounds = array[:, 1]
        return upper_bounds
