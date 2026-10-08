import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from warnings import warn

import numpy as np
import numpy.typing as npt
from scipy import special, stats


@dataclass
class Distribution(ABC):
    mean: float
    cv: float  # coef of variation

    def __post_init__(self) -> None:
        self._derive_parameters()

    @abstractmethod
    def _derive_parameters(self) -> None: ...

    # Derive the distribution paramters from mean and CV,
    # validate the distribution parameters

    @abstractmethod
    def sample(
        self, rng: np.random.Generator, size: int = 1
    ) -> npt.NDArray[np.float64]: ...

    @abstractmethod
    def residual_q(self, q: float, t: float) -> float: ...

    # return the remaining time for a q% chance of arrival,
    # given time t since last arrival
    # i.e. the delay w that satisfies the conditioned probability
    # Pr(X < t+w | X > t) = q


@dataclass
class Exponential(Distribution):
    scale: float = field(init=False)

    def _derive_parameters(self) -> None:
        self.scale = self.mean

        # validate
        if not self.cv == 1:
            raise ValueError(f"@Exponential: cv must be 1, got {self.cv}")
        if not self.scale > 0:
            raise ValueError(f"@Exponential: scale must be > 0, got {self.scale}")

    def sample(
        self, rng: np.random.Generator, size: int = 1
    ) -> npt.NDArray[np.float64]:
        return rng.exponential(scale=self.scale, size=size)

    def residual_q(self, q: float, t: float) -> float:
        return -np.log(1.0 - q) * self.scale


@dataclass
class Erlang(Distribution):
    scale: float = field(init=False)
    k: float = field(init=False)
    APPROXIMATE_QUANTILES: bool = False

    def _derive_parameters(self) -> None:
        self.k = round(1.0 / self.cv**2, 5)
        self.scale = self.mean * self.cv**2

        # validate
        if not self.scale > 0:
            raise ValueError(f"@Erlang: scale must be > 0, got {self.scale}")
        if not self.k.is_integer():
            warn(
                f"@Erlang: k must be integer, got {self.k}. Rounding to {math.ceil(self.k)}"
            )
            self.k = math.ceil(self.k)

    def sample(
        self, rng: np.random.Generator, size: int = 1
    ) -> npt.NDArray[np.float64]:
        return rng.gamma(shape=self.k, scale=self.scale, size=size)

    def residual_q(self, q: float, t: float) -> float:
        if self.APPROXIMATE_QUANTILES:
            # TODO quantile approximation, if required
            return 1.0
        else:
            c = (1 - q) * special.gammaincc(self.k, t / self.scale)
            u = special.gammainccinv(self.k, c)
            return u * self.scale - t


@dataclass
class LogNormal(Distribution):
    mu: float = field(init=False)
    sigma: float = field(init=False)
    APPROXIMATE_QUANTILES: bool = False

    def _derive_parameters(self) -> None:
        self.sigma = math.sqrt(np.log(self.cv**2 + 1))
        self.mu = np.log(self.mean) - 0.5 * np.log(self.cv**2 + 1)

        # validate
        if not self.sigma > 0:
            raise ValueError(f"@LogNormal: sigma must be positive, got {self.sigma}")

    def sample(
        self, rng: np.random.Generator, size: int = 1
    ) -> npt.NDArray[np.float64]:
        return rng.lognormal(mean=self.mean, sigma=self.sigma, size=size)

    def residual_q(self, q: float, t: float) -> float:
        if self.APPROXIMATE_QUANTILES:
            # TODO quantile approximation, if required
            return 1.0
        else:
            return float(
                stats.lognorm.isf(
                    (1 - q)
                    * stats.lognorm.sf(t, s=self.sigma, loc=0, scale=np.exp(self.mu)),
                    s=self.sigma,
                    loc=0,
                    scale=np.exp(self.mu),
                )
                - t
            )
