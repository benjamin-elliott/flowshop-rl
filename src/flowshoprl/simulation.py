import heapq
import itertools
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

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
            raise ValueError(
                f"@SimSpec: All values of S must be greater than or equal to 0."
            )


class event(NamedTuple):
    # declaration order is the sort order for the heap
    time: float
    event_type: int
    prio: int
    epoch: int
    seq: int
    job_id: int


@dataclass(slots=True)
class Job:
    job_id: int  # index in jobs list
    job_class: int  # index into JobClasses tuple in spec
    release: float  # release time into system
    arrived: list[float]  # arrival time at each machine, default [-1.0] * (M+1)
    setup_incurred: list[float]  # setup time incurred at each machine, [0.0] * M


# simulation instance; single spec-to-result object
# simulation structure: makes use of a binary min heap to progress from event to event
# spec defines the simulation paramters
# policy defines the decision policy at each epoch
# eval defines the post-simulation evaluation method, if any
class Simulation:
    def __init__(
        self, rng: np.random.Generator, spec: SimSpec, a_policy: ..., a_eval: ...
    ) -> None:

        # pregenerate arrival time trace for each job class
        self.at_trace = []  # arrival times
        class_rngs = rng.spawn(spec.C)
        for c in range(spec.C):
            self.at_trace.append(
                self._generate_trace(class_rngs[c], spec.job_classes[c].iat, spec.T)
            )

        # generate job list from the arrivals
        merged = sorted((t, c) for c in range(spec.C) for t in self.at_trace[c])
        self.jobs = [
            Job(
                job_id=i,
                job_class=c,
                release=t,
                arrived=[-1.0] * (spec.M + 1),
                setup_incurred=[0.0] * spec.M,
            )
            for i, (t, c) in enumerate(merged)
        ]

        self.seq = itertools.count()
        self.event_heap = []
        for j in self.jobs:
            self.event_heap.append(
                event(
                    time=j.release,
                    event_type=1,
                    prio=j.job_class,
                    epoch=0,
                    seq=next(self.seq),
                    job_id=j.job_id,
                )
            )
        heapq.heapify(self.event_heap)

    # generate an arrival time trace from a specified IAT distribution, truncating at T
    def _generate_trace(
        self, rng: np.random.Generator, iat: dis.Distribution, T: float
    ) -> npt.NDArray[np.float64]:

        # an approximation for the number of samples to generate, should generally loop once
        chunk_size = math.ceil(1.5 * T / iat.mean)
        chunks = []
        t = 0.0
        while True:
            times = t + np.cumsum(iat.sample(rng, chunk_size))
            chunks.append(times)
            if times[-1] > T:
                break
            t = times[-1]

        times = np.concatenate(chunks)
        return times[: np.searchsorted(times, T, side="right")]
