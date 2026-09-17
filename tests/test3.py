import numpy as np
import numpy.random as rand

from flowshoprl.distributions import Exponential
from flowshoprl.evaluators import Flowtime
from flowshoprl.policies import BatchThenWait, ShortestSetup
from flowshoprl.simulation import SimSpec, Simulation
from flowshoprl.structs import JobClass

job_classes = (
    JobClass("a", Exponential(0.6), (0.5, 0.5), 1.0),
    JobClass("b", Exponential(1.0), (0.5, 0.5), 1.0),
    JobClass("c", Exponential(0.4), (0.5, 0.5), 1.0),
)

S = np.array(
    [
        [[0.0, 5.0, 100.0], [50.0, 0.0, 2.0], [80.0, 100.0, 0.0]],
        [[0.0, 1.0, 30.0], [50.0, 0.0, 1.0], [100.0, 100.0, 0.0]],
    ]
)

spec = SimSpec(job_classes, 3, 2, [1.0], 5.0, S)

shortest = ShortestSetup(spec, 1)
batch = BatchThenWait(spec, 1)

I = 1
debug = True
sim1_results = [0.0] * I
sim2_results = [0.0] * I
for i in range(I):
    rng = rand.default_rng(i)
    sim1 = Simulation(rand.default_rng(i), spec, shortest, debug)
    sim2 = Simulation(rand.default_rng(i), spec, batch, debug)

    sim1.simulate()

    sim2.simulate()

    sim1_results[i] = Flowtime(sim1).evaluate()
    sim2_results[i] = Flowtime(sim2).evaluate()

print(max([sim1_results[i] - sim2_results[i] for i in range(len(sim1_results))]))
