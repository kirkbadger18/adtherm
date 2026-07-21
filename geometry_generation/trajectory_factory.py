from ase.atoms import Atoms


class TrajectoryFactory:

    def __init__(self,
                 reference: Atoms,
                 adsorbate_indices: list,
                 ):
        self.reference = reference
        self.indices = adsorbate_indices

    def build_trajectories():
        pass