from abc import ABC, abstractmethod
from random import randint

import numpy as np

from flowshoprl.structs import SimSpec, StateNormalised


class Policy(ABC):
    def __init__(self, spec: SimSpec):
        # K: the number of delay multipliers
        self.K = len(spec.k)
        self.C = len(spec.job_classes)
        self.spec = spec
        self.reset()

    def reset(self) -> None: ...

    @abstractmethod
    def decide(self, state: StateNormalised) -> int: ...

    def encode(self, state:StateNormalised, action_type: int, c, k) -> int:
        match action_type:
            # dispatch
            case 0:return c

            # delay for current class against c, mult k
            case 1: return int(self.C + c * self.K + k)

            # delay until cutoff
            case 2: return int(self.C * (self.K + 1))

        return -1
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
                return self.encode(state, 0, c, None)

        return self.encode(state, 2, None, None)


class BatchThenWait(Policy):
    # dispatch jobs from a class with no setup time
    # if there are no such jobs, wait to the flowtime
    # indifference, for the job with longest setup
    # but only within cutoff.
    # Then, dispatch a job with the shortest setup time
    # among jobs in the queue.
    def reset(self):
        self.last_class = 0

    def decide(self, state: StateNormalised) -> int:
        # dispatch a job with zero setup
        for c in np.where(state.setup == 0)[0]:
            if state.job_queue[c] > 0:
                return self.encode(state, 0, c, None)

        # delay for current setup class against longest setup class
        if state.time < 1:
            c = np.argmax(state.setup)
            return self.encode(state, 1, c, self.K-1)

        else:
            for c in np.argsort(state.setup):
                if state.job_queue[c] > 0:
                    return self.encode(state, 0, c, None)

            return self.encode(state, 2, None, None)
