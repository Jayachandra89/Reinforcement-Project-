"""A lightweight two-phase traffic intersection simulator (no external simulator)."""
import numpy as np

class TrafficEnvironment:
    """Each step lasts one 10-second slot; queues represent vehicles, not seconds.

    Phase 0 serves North/South; phase 1 serves East/West. Arrivals are Poisson.
    Observations are discretized queue lengths (0..5) and current phase.
    """
    def __init__(self, seed=42, horizon=120, arrival_rates=(2.0, 2.5), capacity=5):
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.rates = np.asarray(arrival_rates, dtype=float)
        self.capacity = capacity
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.queues = np.zeros(2, dtype=int)
        self.phase = 0
        self.t = 0
        self.total_vehicle_steps = 0
        self.total_departures = 0
        self.total_switches = 0
        return self.observe()

    def observe(self):
        # Buckets: 0, 1-2, 3-5, 6-9, 10-14, 15+
        def bucket(n): return int(np.searchsorted([1, 3, 6, 10, 15], n, side='right'))
        return (bucket(self.queues[0]), bucket(self.queues[1]), self.phase)

    def step(self, action, arrivals=None):
        if self.t >= self.horizon:
            raise RuntimeError('Episode has ended; call reset().')
        if action not in (0, 1):
            raise ValueError('Action must be 0 (keep) or 1 (switch).')
        switched = int(action == 1)
        if switched:
            self.phase = 1 - self.phase
            self.total_switches += 1
        if arrivals is None:
            new = self.rng.poisson(self.rates)
        else:
            new = np.asarray(arrivals, dtype=int)
            if new.shape != (2,) or np.any(new < 0):
                raise ValueError('arrivals must contain two nonnegative integers')
        self.queues += new
        served = min(self.capacity, int(self.queues[self.phase]))
        self.queues[self.phase] -= served
        self.total_departures += served
        waiting = int(self.queues.sum())
        self.total_vehicle_steps += waiting
        self.t += 1
        # Higher reward for lower queue, small penalty discourages flickering.
        reward = -float(waiting) - 2.0 * switched
        return self.observe(), reward, self.t >= self.horizon, {
            'waiting_vehicles': waiting, 'departures': served, 'arrivals': new.tolist(),
            'switch': bool(switched), 'queues': self.queues.tolist()
        }

    def metrics(self):
        # Little's-law-style aggregate: vehicle-slot waiting / departures.
        return {'mean_queue': self.total_vehicle_steps / self.t if self.t else 0.0,
                'vehicle_slots_waiting': self.total_vehicle_steps,
                'completed_vehicles': self.total_departures,
                'waiting_slots_per_departure': self.total_vehicle_steps / max(1, self.total_departures),
                'switches': self.total_switches, 'remaining_vehicles': int(self.queues.sum())}
