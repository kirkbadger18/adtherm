from ase.atoms import Atoms
from domain import RigidCoordDomain


class CoordinateConverter:

    def __init__(self,
                 reference: Atoms,
                 domain: RigidCoordDomain,
                 ):
        pass

    def to_rigid(self,
                 cartcoords):
        pass

    def to_cartesian(self,
                     rigidcoords):
        pass
