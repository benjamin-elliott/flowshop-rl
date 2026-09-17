from abc import ABC, abstractmethod
from random import randint

import numpy as np

from flowshoprl.structs import StateNormalised


class Policy(ABC):
    def __init__(self, K: int):
        # K: the number of delay multipliers
        self.K = K

    @abstractmethod
    def decide(self, state: StateNormalised) -> int: ...

    # 0:C-1 -> dispatch job of class c
    # C:C*(K+1)-1 -> delay against class c multiplier k
    # C*(K+1) -> delay until cutoff


class Test1(Policy):
    def decide(self, state: StateNormalised) -> int:
        # dispatch lowest index job, otherwise delay until cutoff or delay for first class
        for c, n in enumerate(state.job_queue):
            if n > 0:
                return c

        if state.time < 1:
            return len(state.job_queue) * (self.K + 1)
        else:
            return len(state.job_queue)


class Test2(Policy):
    def decide(self, state: StateNormalised) -> int:
        # select a random decision
        return randint(a=0, b=len(state.job_queue) * (self.K + 1))


class ShortestSetup(Policy):
    # always dispatch jobs from the class with the shortest setup time
    # in the normal case, this means "same class then shortest setup"
    # if two classes share the same setup time, lowest index is taken first
    def decide(self, state: StateNormalised) -> int:
        job_classes = np.argsort(state.setup)
        for c in job_classes:
            if state.job_queue[c] > 0:
                return c

        return len(job_classes) * (self.K + 1)


class BatchThenWaitOnce(Policy):
    # dispatch jobs from a class with no setup time
    # if there are no such jobs, wait to the flowtime
    # indifference, for the job with longest setup
    # but only once in a row.
    # Then, dispatch a job with the shortest setup time
    # among jobs in the queue.
    def __init__(self, K: int):
        self.K = K
        self.can_wait = True
        self.last_class = 0

    def decide(self, state: StateNormalised) -> int:
        for c in np.where(state.setup == 0)[0]:
            if state.job_queue[c] > 0:
                self.can_wait = True
                return int(c)

        if self.can_wait:
            self.can_wait = False
            return int(np.argmax(state.setup) * (self.K + 1) - 1)

        else:
            for c in np.argsort(state.setup):
                if state.job_queue[c] > 0:
                    self.can_wait = True
                    return c

            return len(state.job_queue) * (self.K + 1)
