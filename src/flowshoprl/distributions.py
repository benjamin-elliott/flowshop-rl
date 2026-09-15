from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class Distribution(ABC):
    @abstractmethod
    def sample(
        self, rng: np.random.Generator, size: int = 1
    ) -> npt.NDArray[np.float64]: ...

    @property
    @abstractmethod
    def mean(self) -> float: ...


@dataclass(frozen=True)
class Exponential(Distribution):
    scale: float

    def __post_init__(self):
        if not self.scale > 0:
            raise ValueError(f"@Exponential: scale must be positive, got {self.scale}")

    def sample(
        self, rng: np.random.Generator, size: int = 1
    ) -> npt.NDArray[np.float64]:
        return rng.exponential(scale=self.scale, size=size)

    @property
    def mean(self) -> float:
        return self.scale
