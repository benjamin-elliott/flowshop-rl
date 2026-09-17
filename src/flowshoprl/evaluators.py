from abc import ABC, abstractmethod

from flowshoprl.simulation import Simulation


class Evaluator[T](ABC):
    def __init__(self, sim: Simulation) -> None:
        self.sim = sim

    @abstractmethod
    def evaluate(self) -> T: ...

    @abstractmethod
    def display(self) -> None: ...

    @property
    def _objective(self) -> T: ...


class Makespan(Evaluator[float]):
    def evaluate(self) -> float:
        return self.sim.state.time

    def display(self) -> None:
        print("\n" + "*" * 40)
        print(f"Makespan: {self._objective}")
        print("*" * 40)

    @property
    def _objective(self) -> float:
        return self.sim.state.time


class Flowtime(Evaluator[float]):
    def evaluate(self) -> float:
        self.flow = 0.0
        for job in self.sim.jobs:
            c = job.job_class
            self.flow += self.sim.spec.job_classes[c].weight * (job.arrived[-1] - job.release)
        
        return self.flow

    def display(self) -> None:
        _ = self.evaluate()
        print('\n' + "*"*40)
        print(f'Weighted Flowtime: {self.flow}')
        print('\n' + "*"*40)

    @property
    def _objective(self) -> float:
        return self.evaluate()
