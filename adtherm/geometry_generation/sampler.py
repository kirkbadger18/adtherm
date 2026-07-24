from .domain import RigidCoordDomain
from scipy.stats import qmc


class BaseSampler:

    def __init__(self,
                 domain: RigidCoordDomain,

                 ):
        self.domain = domain


class SobolSampler(BaseSampler):

    def __init__(self, domain):
        super().__init__(domain)

    def draw_samples(self, N):
        sobol_engine = qmc.Sobol(d=self.domain.N_dimensions,
                                 scramble=True,
                                 seed=42,
                                 )
        samples = sobol_engine.random(n=N)
        lower_bounds = self.domain.lower_bounds()
        upper_bounds = self.domain.upper_bounds()
        scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
        return scaled_samples
