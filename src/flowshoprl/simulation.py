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
        self,
        rng: np.random.Generator,
        spec: SimSpec,
        a_policy: Policy,
        debug: bool = False,
    ) -> None:

        self.spec = spec
        self.policy = a_policy
        self.policy.reset()
        self.m0_free = True
        self.setup_class = 0  # default the setup state
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
                due=t + spec.job_classes[c].due_offset,
                arrived=[-1.0] * (spec.M + 1),
                setup_incurred=[0.0] * spec.M,
            )
            for i, (t, c) in enumerate(merged)
        ]
        self.remaining_jobs = len(self.jobs)

        # initialise event heap, and populate initial events from job arrivals and due dates
        self.seq = itertools.count()
        self.epoch = 0
        self.event_heap = []
        for j in self.jobs:
            # arrivals
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

            # due dates
            self.event_heap.append(
                Event(
                    time=j.due,
                    event_type=3,
                    prio=j.job_class,
                    epoch=self.epoch,
                    seq=next(self.seq),
                    job_id=j.job_id,
                )
            )

        heapq.heapify(self.event_heap)

        # populate initial state:
        self.state = State(
            0.0, self.spec.S[0, 0, :], [0] * spec.C, np.array([0.0] * spec.M)
        )

        # populate initial cost and tracking metrics
        self.jobs_in_system = 0

        self.accrued_time_in_system = 0.0
        self._last_accrued_time_in_system = 0.0

        self.late_penalty = 0.0
        self._last_late_penalty = 0.0

        self._time_at_last_event = 0.0

        # populate debug fields
        self.debug = debug
        if self.debug:
            self.decision_log = []

    def simulate(self):
        while self.remaining_jobs:
            self._step_event()

        if self.debug:
            print(self.state)
            print("\n")
            print("*" * 50)
            print("\n")

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

        # accrue time in system:
        self.accrued_time_in_system += self.jobs_in_system * dt

        if event.time < current.time:
            raise RuntimeError(
                f"Time ran backwards. Popped {event.time}, current time is {current.time}. Offending event was: {event}"
            )

        if self.debug:
            print("\n")

        match event.event_type:
            # 0: operation completion
            case 0:
                if self.debug:
                    print(f"[OPERATION COMPLETE: M{-event.prio} J{event.job_id}]")

                # last machine
                if event.prio == -(self.spec.M - 1):
                    self.remaining_jobs -= 1
                    self.jobs_in_system -= 1
                    self.jobs[event.job_id].is_complete = True

                # first machine
                if event.prio == 0:
                    self.m0_free = True

                self.state = State(
                    event.time,
                    current.setup,
                    current.job_queue,
                    np.array([max(0.0, time - dt) for time in current.drain_time]),
                )

            # 1: job arrival
            case 1:
                if self.debug:
                    print(f"[JOB ARRIVAL: C:{event.prio} J{event.job_id}]")

                # increment the job queue
                queue_increment = [0] * self.spec.C
                queue_increment[event.prio] = 1
                self.state = State(
                    event.time,
                    current.setup,
                    [c + i for c, i in zip(current.job_queue, queue_increment)],
                    np.array([max(0.0, time - dt) for time in current.drain_time]),
                )

                self.jobs_in_system += 1

                # add the job id to the relevant deque
                self.job_ids[event.prio].append(event.job_id)

            # 2: wait/delay passed
            case 2:
                if self.debug:
                    print(f"[WAIT END]")

                if event.epoch == self.epoch:
                    self.state = State(
                        event.time,
                        current.setup,
                        current.job_queue,
                        np.array([max(0.0, time - dt) for time in current.drain_time]),
                    )

                else:
                    if self.debug:
                        print("\n")
                        print(event)
                        print("Stale delay -- skipping...")

                    return

            # 3: due date reached
            case 3:
                if self.debug:
                    print(f"[DUE DATE]")

                if self.jobs[event.job_id].is_complete:
                    if self.debug:
                        print("\n")
                        print(event)
                        print("Job has already been completed -- skipping...")

                    return

                else:
                    self.late_penalty += 1 / len(self.jobs)

        # once state is updated, check if a decision should be made
        # call policy function and insert new events if so
        if self.debug:

            print(event)
            print(f"Jobs in system: {self.jobs_in_system}")
            print(f"Remaining jobs: {self.remaining_jobs}/{len(self.jobs)}")
            print(f"Deicison ready? {self._decision_ready}")
            print(f"Epoch: {self.epoch}")
            print(f"Drain time: {self.state.drain_time}")
            print(f"Job queue: {self.state.job_queue}")

            # objective metrics
            print(f"Total time in system = {self.accrued_time_in_system:.4f}")
            print(f"Lateness penalty     = {self.late_penalty:.4f}")

        if self._decision_ready:
            if self.debug:
                print("\n")
                print("Last epoch accrual:")
                print(
                    f"Total time in system:{self.accrued_time_in_system - self._last_accrued_time_in_system:.4f}"
                )
                print(
                    f"Late penalty: {self.late_penalty - self._last_late_penalty:.4f}"
                )

            # update matric tracking
            self._last_accrued_time_in_system = self.accrued_time_in_system
            self._last_late_penalty = self.late_penalty

            # advance the epoch and make a new decision
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
            print(f"Taking action {action} at {current.time}")
        events = []

        if action <= self.spec.C - 1:
            # insert operation completion events; decrement job queue counter; update setup class
            c = action

            if self.job_ids[c]:
                j = self.job_ids[c].popleft()
                self.m0_free = False

                t = np.array([0.0] * self.spec.M)
                t[0] = (
                    self.state.drain_time[0]
                    + self.state.setup[c]
                    + self.spec.job_classes[c].proc_times[0]
                )
                events.append(
                    Event(current.time + t[0], 0, 0, self.epoch, next(self.seq), j)
                )

                self.jobs[j].arrived[0] = current.time
                self.jobs[j].setup_incurred[0] = self.state.setup[c]

                # TODO: we don't actually need to consider events where non m0 machines are released - just write completions directly to jobs
                for m in range(1, self.spec.M):
                    t[m] = (
                        max(self.state.drain_time[m], t[m - 1])
                        + self.spec.S[m][self.setup_class][c]
                        + self.spec.job_classes[c].proc_times[m]
                    )
                    events.append(
                        Event(current.time + t[m], 0, -m, self.epoch, next(self.seq), j)
                    )
                    
                    self.jobs[j].arrived[m] = current.time + t[m-1]
                    self.jobs[j].setup_incurred[m] = self.spec.S[m][self.setup_class][c]

                self.jobs[j].arrived[-1] = current.time + t[-1]

                queue_decrement = [0] * self.spec.C
                queue_decrement[c] = 1

                self.state = State(
                    current.time,
                    self.spec.S[0][c][:],
                    [c - d for c, d in zip(current.job_queue, queue_decrement)],
                    t,
                )

                self.setup_class = c

                if self.debug:
                    print(f"Action taken: class {c} (job {j}) dispatched")

            elif self.debug:
                print(
                    f"Action {action} invalid -- class {c} has no jobs waiting to dispatch. Skipping..."
                )

        elif action <= self.spec.C * (len(self.spec.k) + 1) - 1:
            # insert delay action against specified job class
            # extract the class index and mult index
            idx = action - self.spec.C
            c = idx // len(self.spec.k)
            k = idx % len(self.spec.k)
            delay = (
                0.5
                * self.spec.k[k]
                * (
                    sum(self.spec.S[:, self.setup_class, c])
                    + sum(self.spec.job_classes[c].proc_times[:])
                    + sum(self.spec.S[:, c, self.setup_class])
                    - sum(self.spec.job_classes[self.setup_class].proc_times[:])
                )
            )
            if self.debug:
                print(f'Waiting for class {self.setup_class} against class {c}. Attempted delay: {delay}')

            if delay > 0:
                events.append(
                    Event(current.time + delay, 2, 0, self.epoch, next(self.seq), -1)
                )
                if self.debug:
                    print(f"Action taken: delayed for class {self.setup_class} against {c}. ({delay})")

        elif action == self.spec.C * (len(self.spec.k) + 1):
            # insert delay action until cutoff time
            if self.spec.T > current.time:
                events.append(Event(self.spec.T, 2, 0, self.epoch, next(self.seq), -1))
                if self.debug:
                    print(
                        f"Action taken: delayed for class {self.setup_class} to the cutoff. ({self.spec.T - current.time})"
                    )

        else:
            raise RuntimeError(
                f"Specified action is invalid. Expected an integer in the range [0, {self.spec.C*(len(self.spec.k)+1)}, got {action!r}]"
            )

        # TODO remove this at some point, and handle the empty event heap as a runtime error
        # TODO once policy masking has been properly implemented
        if not events:
            raise RuntimeError(f'Decision at {current.time} made no events. All decisions must result in event insertion to prevent hanging.')
        for event in events:
            heapq.heappush(self.event_heap, event)

    @property
    def normalised_state(self) -> StateNormalised:
        current = self.state
        return StateNormalised(
            time=current.time / self.spec.T,
            setup=current.setup / self.spec.mpt,
            job_queue=current.job_queue,
            drain_time=current.drain_time / self.spec.mpt,
        )
