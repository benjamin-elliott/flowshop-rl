from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Distribution(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float: ...

    @property
    @abstractmethod
    def mean(self) -> float: ...


@dataclass(frozen=True)
class Exponential(Distribution):
    scale: float

    def __post_init__(self):
        if not self.scale > 0:
            raise ValueError(f"@Exponential: scale must be positive, got {self.scale}")

    def sample(self, rng: np.random.Generator) -> float:
        return rng.exponential(scale=self.scale)

    @property
    def mean(self) -> float:
        return self.scale
