from ase.atoms import Atoms
from domain import RigidCoordDomain
from sampler import SobolSampler
from coordinates import CoordinateConverter
import numpy as np


class Generator:
    """
    This class handles generating a list of ASE atoms objects which need to be
    evaluated with DFT and are used by the AdTherm workflow to train a
    surrogate energy surface.

    A Generator object is instantiated using a list of ASE Atoms objects
    that contain the coordinates and potential energies at the minima.
    It takes the 3N hessians for each minima as well, which can be
    obtained using ASE and the VibtrationsData object. Lastly the indices of
    the adsorbate need to be specified.
    """

    def __init__(self,
                 minima: list[Atoms],
                 hessians_3N: list[np.ndarray],
                 adsorbate_indices: list,
                 ):
        self.minima = minima
        self.Hessians_3N = hessians_3N
        self.adsorbate_indices = adsorbate_indices
        self.rigid_domain = RigidCoordDomain(self.adsorbate_indices)
        self.coord_converter = CoordinateConverter(self.minima[0],
                                                   self.rigid_domain,
                                                   )
    
    def generate_gaussian_samples(self,
                                  N: int,
                                  T: float,
                                  ) -> list[Atoms]:
        """
        This function generates a list of ASE atoms objects which are intended
        to be fed to a calculator to be evaluated. The samples are generated
        using a Gaussian distribution which is controlled by the temperature
        and the covariance matric which is computed using the rigid-body
        hessians in their fixed frame coordinates.
        See: {insert Badger2026 paper}
        N is the number of samples drawn per minima.
        T is the temperature, which controlls gaussian width. Since the systems
        studied are anharmonic, one should sample well above and below their
        actual target temperature.
        """
        pass

    def generate_sobol_samples(self,
                               N: int,
                               ) -> list[Atoms]:
        """"
        This function generates a list of ASE atoms objects which are intended
        to be fed to a calculator to be evaluated. The samples are generated
        using a Sobol sequence. N is the number of samples drawn.
        """
        sampler = SobolSampler(self.rigid_domain)
        rigidcoords = sampler.draw_samples(N)
        cartcoords = self.coord_converter.to_cartesian(rigidcoords)
        trajectories = self.traj_factory.build_trajectories(cartcoords)
        return trajectories

    def generate_random_samples(self,
                                N: int,
                                ) -> list[Atoms]:
        """"
        This function generates a list of ASE atoms objects which are intended
        to be fed to a calculator to be evaluated. The samples are generated
        using a uniform random distribution. N is the number of samples drawn.
        """
        pass


class AutoGenerator(Generator):
    """
    This class is to be used as a defaults class that generates sameples in a
    way that should be suffigient for downstream training.
    """
    def __init__(self, minima, hessians_3N, adsorbate_indices):
        super().__init__(minima, hessians_3N, adsorbate_indices)

    def generate_samples(self):
        """
        The main function of this class, it is used to generate a standard set
        of training data assumed to be generally learnable.
        """
        pass
