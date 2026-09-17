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
        print(f"Makespan: {self.sim.state.time}")
        print("*" * 40)

    @property
    def _objective(self) -> float:
        return self.sim.state.time
