import numpy as np
import numpy.random as rand

from flowshoprl.distributions import Exponential
from flowshoprl.evaluators import Flowtime, Makespan
from flowshoprl.policies import BatchThenWait, ShortestSetup
from flowshoprl.simulation import SimSpec, Simulation
from flowshoprl.structs import JobClass

job_classes = (
    JobClass("a", Exponential(2.0), (1.5, 0.5), 1.0),
    JobClass("b", Exponential(3.0), (0.5, 2.5), 1.0),
    JobClass("c", Exponential(3.0), (2.0, 1.0), 1.0),
)

S = np.array(
    [
        [[0.0, 5.0, 1.0], [5.0, 0.0, 2.0], [8.0, 1.0, 0.0]],
        [[0.0, 1.0, 3.0], [5.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    ]
)

spec = SimSpec(job_classes, 3, 2, [1.0], 100.0, S)
print(spec.load.regime)

shortest = ShortestSetup(spec)
batch = BatchThenWait(spec)

I = 10
debug = False
sim1_results = [0.0] * I
sim2_results = [0.0] * I
sim1_makespan = [0.0] * I
sim2_makespan = [0.0] * I
for i in range(I):
    rng = rand.default_rng(i)
    sim1 = Simulation(rand.default_rng(i), spec, shortest, debug)
    sim2 = Simulation(rand.default_rng(i), spec, batch, debug)

    sim1.simulate()

    sim2.simulate()

    sim1_results[i] = Flowtime(sim1).evaluate()
    sim2_results[i] = Flowtime(sim2).evaluate()

    sim1_makespan[i] = Makespan(sim1).evaluate()
    sim2_makespan[i] = Makespan(sim2).evaluate()

print('Comparison: ShortestFirst vs BatchThenWait, weighted Flowtime')
print(f'Max Delta (SF-BW) = {max([sim1_results[i] - sim2_results[i] for i in range(I)])}')
print(f'Min Delta (SF-BW) = {min([sim1_results[i] - sim2_results[i] for i in range(I)])}')
print(f'Avg Delta (SF-BW) = {np.mean([sim1_results[i] - sim2_results[i] for i in range(I)])}')

print('Comparison: ShortestFirst vs BatchThenWait, Makespan')
print(f'Max Delta (SF-BW) = {max([sim1_makespan[i] - sim2_makespan[i] for i in range(I)])}')
print(f'Min Delta (SF-BW) = {min([sim1_makespan[i] - sim2_makespan[i] for i in range(I)])}')
print(f'Avg Delta (SF-BW) = {np.mean([sim1_makespan[i] - sim2_makespan[i] for i in range(I)])}')

