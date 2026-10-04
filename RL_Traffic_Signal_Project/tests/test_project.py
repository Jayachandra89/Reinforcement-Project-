import unittest
import numpy as np
from traffic_environment import TrafficEnvironment
from q_learning_agent import QLearningAgent
from fixed_timer import fixed_timer_action

class ProjectTests(unittest.TestCase):
    def test_queue_and_reward(self):
        e = TrafficEnvironment(horizon=2)
        self.assertEqual(e.reset(), (0, 0, 0))
        state, reward, done, info = e.step(0, arrivals=[4, 2])
        self.assertEqual(info['queues'], [0, 2]); self.assertEqual(reward, -2.0)
        self.assertFalse(done)
        state, reward, done, info = e.step(1, arrivals=[0, 1])
        self.assertEqual(info['queues'], [0, 0]); self.assertEqual(reward, -2.0)
        self.assertTrue(done)
    def test_seed_reproducibility(self):
        a, b = TrafficEnvironment(seed=4), TrafficEnvironment(seed=4)
        for _ in range(12):
            self.assertEqual(a.step(0)[3]['arrivals'], b.step(0)[3]['arrivals'])
    def test_q_update_terminal(self):
        a = QLearningAgent(alpha=1.0)
        a.update((0,0,0), 1, -5, (1,1,1), True)
        self.assertEqual(a.q[0,0,0,1], -5)
    def test_baseline(self):
        self.assertEqual([fixed_timer_action(x) for x in range(7)], [0,0,0,1,0,0,1])

if __name__ == '__main__': unittest.main()
