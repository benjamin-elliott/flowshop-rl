from abc import ABC, abstractmethod
from random import randint

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
