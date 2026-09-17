import numpy as np
import numpy.random as rand

from flowshoprl.distributions import Exponential
from flowshoprl.policies import Test2
from flowshoprl.simulation import SimSpec, Simulation
from flowshoprl.structs import JobClass

job_classes = (
        JobClass('a', Exponential(1.0), (1.0, 2.0), 1.0),
        JobClass('b', Exponential(2.0), (2.0, 1.0), 1.0)
        )

S = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]]
            ]
        )

spec = SimSpec(job_classes, 2, 2, [1.0], 10.0, S)
rng = rand.default_rng(42)
policy = Test2(1)

sim = Simulation(rng, spec, policy, None, True)
sim.simulate()
