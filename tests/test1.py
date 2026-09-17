from flowshoprl.simulation import Simulation, SimSpec
from flowshoprl.structs import JobClass
from flowshoprl.distributions import Exponential
import numpy as np
import numpy.random as rand
from flowshoprl.policies import Test1   

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
policy = Test1(1)

sim = Simulation(rng, spec, policy, None, True)
sim.simulate()
