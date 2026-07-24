from ase.atoms import Atoms
import numpy as np
from .coordinates import CoordinateConverter


class RigidMinima:

    def __init__(self,
                 minima: list[Atoms],
                 hessians_3N: list[np.ndarray],
                 coordinate_converter: CoordinateConverter,
                 ):

        self.minima = minima
        self.hessians_3N = hessians_3N
        self.coordinate_converter = coordinate_converter
        self._assign_rigid_minima()
        self._assign_rigid_hessians()

    def _assign_rigid_minima(self):
        self.rigid_minima = None

    def _assign_rigid_hessians(self):
        self.rigid_hessians = None
