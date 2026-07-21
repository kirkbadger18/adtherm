from .domain import RigidCoordDomain


class BaseSampler:

    def __init__(self,
                 domain: RigidCoordDomain,
                 ):
        self.domain = domain


class SobolSampler(BaseSampler):

    def __init__(self, domain):
        super().__init__(domain)

    def draw_samples(self, N):
        pass
