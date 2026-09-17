from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from flowshoprl.distributions import Distribution


class Event(NamedTuple):
    # declaration order is the sort order for the heap
    # event_type: 0/1/2 -> operation completion/job arrival/wait complete
    # prio: depending on event_type -> -1*machine/job class/0
    # epoch: the decision epoch as of event creation, used to cancel wait events
    # seq: unique insertion counter for tie breaking
    # job id: integer index of the relevant job in the jobs list, if event type 0/1
    time: float
    event_type: int
    prio: int
    epoch: int
    seq: int
    job_id: int


class State(NamedTuple):
    # (time, m0 setup row vector, count of jobs in queue, drain time per machine)
    time: float
    setup: npt.NDArray[np.float64]
    job_queue: list[int]
    drain_time: npt.NDArray[np.float64]


@dataclass(slots=True)
class Job:
    job_id: int  # index in jobs list
    job_class: int  # index into JobClasses tuple in spec
    release: float  # release time into system
    arrived: list[
        float
    ]  # arrival time at each machine (and M departure time), default [-1.0] * (M+1)
    setup_incurred: list[float]  # setup time incurred at each machine, [0.0] * M


@dataclass(frozen=True)
class JobClass:
    name: str
    iat: Distribution
    proc_times: tuple[float, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        for idx, pt in enumerate(self.proc_times):
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


class Regime(Enum):
    # whether the offered load leaves the policy any room to keep up
    SATURATED = "saturated for any policy"
    CONTESTED = "policy-dependent: batching decides stability"
    STABLE = "stable for any policy"


class LoadProfile(NamedTuple):
    # expected work arriving per unit time, per machine
    # processing excludes setup; lower/upper add the strict setup bounds
    processing: npt.NDArray[np.float64]
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]

    @property
    def bottleneck(self) -> int:
        # the machine that saturates first
        return int(np.argmax(self.upper))

    @property
    def regime(self) -> Regime:
        if self.lower.max() >= 1.0:
            return Regime.SATURATED
        if self.upper.max() < 1.0:
            return Regime.STABLE
        return Regime.CONTESTED


# simulation spec; specify sim parameters before RNG
@dataclass
class SimSpec:
    job_classes: tuple[JobClass, ...]
    C: int  # number of classes
    M: int  # number of machines
    k: list[float]  # set of delay multipliers
    T: float  # cutoff period for orders
    S: np.ndarray[tuple[int, int, int]]  # setup times matrix: M,j,i

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
            raise ValueError(
                f"@SimSpec: All values of S must be greater than or equal to 0."
            )

        self.mpt = 0.0
        n = 0
        for job_class in self.job_classes:
            for p in job_class.proc_times:
                self.mpt += p
                n += 1

        self.mpt = self.mpt / n

        S = np.array(self.S, dtype=float, copy=True)
        S.setflags(write=False)
        self.S = S

    @property
    def arrival_rates(self) -> npt.NDArray[np.float64]:
        # expected arrivals per unit time, per class
        return np.array([job_class.iat.mean for job_class in self.job_classes]) ** -1.0

    @property
    def load(self) -> LoadProfile:
        # expected offered work per unit time, per machine, with strict setup bounds.
        # every class c dispatch pays S[m][j][c] for whichever class j preceded it,
        # so it costs between the min and max of that column whatever the policy does.
        # exact given expected arrival counts; a single trace varies with the number sampled.
        rates = self.arrival_rates
        processing = np.zeros(self.M)
        setup_lo = np.zeros(self.M)
        setup_hi = np.zeros(self.M)
        for c, job_class in enumerate(self.job_classes):
            processing += rates[c] * np.asarray(job_class.proc_times)
            col = self.S[:, :, c]  # (M, C): setup into c from every class
            setup_lo += rates[c] * col.min(axis=1)
            setup_hi += rates[c] * col.max(axis=1)

        return LoadProfile(processing, processing + setup_lo, processing + setup_hi)


class StateNormalised(NamedTuple):
    # State, but normalised.
    # time is normalised as a fraction of the total time before cutoff. Exceeds 1 during cleanup
    # setup and processing times are normalised as a fraction of the mean processing time
    time: float
    setup: npt.NDArray[np.float64]
    job_queue: list[int]
    drain_time: npt.NDArray[np.float64]
