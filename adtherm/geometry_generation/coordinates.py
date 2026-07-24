from ase.atoms import Atoms
import numpy as np


class AdsorbateReference:
    def __init__(self,
                 reference_atoms: Atoms,
                 adsorbate_indices: list,
                 ):
        self.atoms = reference_atoms
        self.ads_indices = adsorbate_indices
        self.adsorbate = reference_atoms[adsorbate_indices].copy()
        self._evaluate_reference()

    def _evaluate_reference(self):
        self.com = self.adsorbate.get_center_of_mass()
        self.positions = self.adsorbate.positions - self.com
        pa = self.adsorbate.get_moments_of_inertia(vectors=True)
        self.principle_axis = pa


class CoordinateConverter:
    def __init__(self,
                 reference: AdsorbateReference,
                 ):
        self.reference = reference

    def _lattice_vecs_to_xy(self, v1, v2):
        pass

    def _xy_to_lattice_vecs(self, x, y):
        pass

    def to_cartesian(self, rigid_coords):
        cart_coord_list = []
        for i, rigid_coord in enumerate(rigid_coords):
            cart_coord = self.reference.positions.copy()
            rot_mat = self.get_rotation_matrix(rigid_coords[2::])
            cart_coord = np.matmul(rot_mat, cart_coord)
            x, y = self._lattice_vecs_to_xy(rigid_coord[0], rigid_coord[1])
            xyz = np.array([x, y, cart_coord[2]])
            cart_coord[:, 2] += self.reference.com + xyz
            cart_coord_list.append(cart_coord)
        return cart_coord_list

    def to_rigid(self, cartsian_coords):
        pass
