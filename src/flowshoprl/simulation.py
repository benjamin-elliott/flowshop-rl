from dataclasses import dataclass

import numpy as np

import flowshoprl.distributions as dis


# defines a class of jobs
# iat: Inter-arrival time distribution
# proc_times: per-machine deterministic processing times
# weight: objective weight; defaults to 1.0
@dataclass(frozen=True)
class JobClass:
    name: str
    iat: dis.Distribution
    proc_times: tuple[float]
    weight: float = 1.0

    def __post_init__(self) -> None:
        for pt, idx in enumerate(self.proc_times):
            if not pt > 0:
                raise ValueError(
                    f"@JobClass: ({self.name}) all processing times must be greater than 0. Got {pt} at index {idx}"
                )

        if not self.weight > 0:
            raise ValueError(
                f"@JobClass: ({self.name}) weight must be greater than 0. Got {self.weight}"
            )

    @property
    def num_machines(self) -> int:
        return len(self.proc_times)


# simulation spec; specify sim parameters before RNG
@dataclass(frozen=True)
class SimSpec:
    job_classes: tuple[JobClass, ...]
    C: int  # number of classes
    M: int  # number of machines
    T: float  # cutoff period for orders
    S: np.ndarray[tuple[int, int, int]]  # setup times matrix

    def __post_init__(self) -> None:
        if not self.C > 0:
            raise ValueError(f"@SimSpec: C must be greater than 0. Got {self.C}")

        if not self.M > 0:
            raise ValueError(f"@SimSpec: M must be greater than 0. Got {self.M}")

        if not len(self.job_classes) == self.C:
            raise ValueError(
                f"@SimSpec Number of job classes must match C. Got {len(self.job_classes)}, wanted {self.C}"
            )

        for job_class in self.job_classes:
            if not job_class.num_machines == self.M:
                raise ValueError(
                    f"@SimSpec: ({job_class.name}) Processing times vector must have exactly {self.M} entries. Got {job_class.num_machines}"
                )

        if not self.T > 0:
            raise ValueError(f"@SimSpec: T must be greater than 0. Got {self.T}")

        if not np.all(self.S >= 0):
            raise ValueError(f"@SimSpec: All values of S must be greater than 0.")
