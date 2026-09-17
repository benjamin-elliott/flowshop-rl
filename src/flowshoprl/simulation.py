import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import flowshoprl.distributions as dis
from flowshoprl.policies import Policy
from flowshoprl.structs import (Event, Job, JobClass, SimSpec, State,
                                StateNormalised)


# simulation instance; single spec-to-result object
# simulation structure: makes use of a binary min heap to progress from event to event
# spec defines the simulation paramters
# policy defines the decision policy at each epoch
# eval defines the post-simulation evaluation method, if any
class Simulation:
    def __init__(
            self, rng: np.random.Generator, spec: SimSpec, a_policy: Policy, a_eval: ..., debug: bool = False
    ) -> None:

        self.spec = spec
        self.policy = a_policy
        self.evaluator = a_eval
        self.m0_free = True
        self.setup_class = 0 # default the setup state
        self.job_ids = [deque() for _ in range(self.spec.C)]

        # pregenerate arrival time trace for each job class
        # no need to keep the rng; all values are pregenerated
        at_trace = []  # arrival times
        class_rngs = rng.spawn(spec.C)
        for c in range(spec.C):
            at_trace.append(
                self._generate_trace(class_rngs[c], spec.job_classes[c].iat, spec.T)
            )

        # generate job list from the arrivals
        merged = sorted((t, c) for c in range(spec.C) for t in at_trace[c])
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
        self.remaining_jobs = len(self.jobs)

        # initialise event heap, and populate initial events from job arrivals
        self.seq = itertools.count()
        self.epoch = 0
        self.event_heap = []
        for j in self.jobs:
            self.event_heap.append(
                Event(
                    time=j.release,
                    event_type=1,
                    prio=j.job_class,
                    epoch=self.epoch,
                    seq=next(self.seq),
                    job_id=j.job_id,
                )
            )
        heapq.heapify(self.event_heap)

        # populate initial state:
        self.state = State(0.0, self.spec.S[0,0,:], [0]*spec.C, np.array([0.0]*spec.M))

        # populate debug fields
        self.debug = debug 
        if self.debug:
            self.decision_log = []

    def simulate(self):
        while self.remaining_jobs:
            self._step_event()

        print(self.state)

    # generate an arrival time trace from a specified IAT distribution, truncating at T
    def _generate_trace(
        self, rng: np.random.Generator, iat: dis.Distribution, T: float
    ) -> npt.NDArray[np.float64]:

        # an approximation for the number of samples to generate, should generally loop once
        chunk_size = math.ceil(T / iat.mean)
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

    def _step_event(self) -> None:
        current = self.state
        event = heapq.heappop(self.event_heap)
        dt = event.time - current.time

        if event.time < current.time:
            raise RuntimeError(f'Time ran backwards. Popped {event.time}, current time is {current.time}. Offending event was: {event}')

        match event.event_type:
            # 0: operation completion
            case 0:
                if event.prio == -(self.spec.M - 1):
                    self.remaining_jobs -= 1

                if event.prio == 0:
                    self.m0_free = True

                self.state = State(event.time, current.setup, current.job_queue, np.array([max(0.0, time - dt) for time in current.drain_time]))

            # 1: job arrival
            case 1:
                # increment the job queue
                queue_increment = [0] * self.spec.C
                queue_increment[event.prio] = 1
                self.state = State(event.time, current.setup, [c + i for c,i in zip(current.job_queue, queue_increment)], np.array([max(0.0, time - dt) for time in current.drain_time]))

                # add the job id to the relevant deque
                self.job_ids[event.prio].append(event.job_id)

            # 2: wait/delay passed
            case 2:
                if event.epoch == self.epoch:
                    self.state = State(event.time, current.setup, current.job_queue, np.array([max(0.0, time - dt) for time in current.drain_time]))

                else:
                    if self.debug:
                        print('\n' + '='*10)
                        print(event)
                        print('Stale delay -- skipping...')

                    return

        # once state is updated, check if a decision should be made
        # call policy function and insert new events if so
        if self.debug:

            print('\n' + '='*10)
            print(event)
            print(f'Remaining jobs: {self.remaining_jobs}')

            print(f'Deicison ready? {self._decision_ready}')
            print(f'Drain time: {self.state.drain_time}')
            print(f'Job queue: {self.state.job_queue}')

        if self._decision_ready:
            self.epoch += 1
            self.decode_action(self.policy.decide(self.normalised_state), self.state)

    @property 
    def _decision_ready(self) -> bool:
        # true if the current state is a decision epoch, and a deicison has not yet been made
        # decision epoch: machine 0 free, at least one job in queue
        return self.m0_free and sum(self.state.job_queue) > 0


    def decode_action(self, action: int, current: State) -> None:
        # decode the action integer into a series of event insertions
        # 0:C-1 -> dispatch job of class c
        # C:C*(K+1)-1 -> delay against class c multiplier k
        # C*(K+1) -> delay until cutoff
        if self.debug:
            print(f'Taking action {action} at {current.time}')
        events = []

        if action <= self.spec.C - 1:
            # insert operation completion events; decrement job queue counter; update setup class
            c = action

            if self.job_ids[c]:
                j = self.job_ids[c].popleft()
                self.m0_free = False

                t = np.array([0.0] * self.spec.M)
                t[0] = self.state.drain_time[0] + self.state.setup[c] + self.spec.job_classes[c].proc_times[0]
                events.append(Event(current.time + t[0], 0, 0, self.epoch, next(self.seq), j))

                #TODO: we don't actually need to consider events where non m0 machines are released - just write completions directly to jobs
                for m in range(1, self.spec.M):
                    t[m] = max(self.state.drain_time[m], t[m-1]) + self.spec.S[m][self.setup_class][c] + self.spec.job_classes[c].proc_times[m]
                    events.append(Event(current.time + t[m], 0, -m, self.epoch, next(self.seq), j))

                queue_decrement = [0]*self.spec.C
                queue_decrement[c] = 1

                self.state = State(current.time, self.spec.S[0][c][:], [c - d for c,d in zip(current.job_queue, queue_decrement)], t)

                self.setup_class = c

                if self.debug:
                    print(f'Action taken: class {c} (job {j}) dispatched')

            elif self.debug:
                print(f'Action {action} invalid -- class {c} has no jobs waiting to dispatch. Skipping...')

        elif action <= self.spec.C * ( len(self.spec.k) + 1) - 1:
            # insert delay action against specified job class
            # extract the class index and mult index
            idx = action - self.spec.C
            c = idx // len(self.spec.k)
            k = idx % len(self.spec.k)
            delay = .5 * self.spec.k[k] * (self.state.setup[c] + self.spec.job_classes[c].proc_times[0] + self.spec.S[0,c,self.setup_class] - self.spec.job_classes[self.setup_class].proc_times[0])
            if delay > 0:
                events.append(Event(current.time + delay, 2, 0, self.epoch, next(self.seq), -1))
                if self.debug:
                    print(f'Action taken: delayed for class {c} ({delay})')

        elif action == self.spec.C * ( len(self.spec.k) + 1 ):
            # insert delay action until cutoff time
            if  self.spec.T > current.time:
                events.append(Event(self.spec.T, 2, 0, self.epoch, next(self.seq), -1))
                if self.debug:
                    print(f'Action taken: delayed for cutoff ({self.spec.T - current.time})')

        else: 
            raise RuntimeError(f'Specified action is invalid. Expected an integer in the range [0, {self.spec.C*(len(self.spec.k)+1)}, got {action!r}]')

        #TODO remove this at some point, and handle the empty event heap as a runtime error
        #TODO once policy masking has been properly implemented
        if not events:
            events.append(Event(current.time, 2, 0, self.epoch, next(self.seq), -1))
        for event in events:
            heapq.heappush(self.event_heap, event)

    @property
    def normalised_state(self) -> StateNormalised:
        current = self.state
        return StateNormalised(
                time=current.time/self.spec.T,
                setup=current.setup/self.spec.mpt,
                job_queue=current.job_queue,
                drain_time=current.drain_time/self.spec.mpt
                )
