"""Train, evaluate, and plot traffic-control policies.

Run: python main.py --episodes 1500
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from traffic_environment import TrafficEnvironment
from q_learning_agent import QLearningAgent
from fixed_timer import fixed_timer_action

OUT = Path(__file__).resolve().parent / 'results'

def train(episodes=1500, seed=42):
    agent = QLearningAgent(seed=seed + 1)
    rewards = []
    for ep in range(episodes):
        env = TrafficEnvironment(seed=seed + ep)
        state = env.reset()
        total = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state, total = next_state, total + reward
        agent.decay()
        rewards.append(total)
    return agent, rewards

def evaluate(agent, seeds, policy):
    records = []
    for seed in seeds:
        env = TrafficEnvironment(seed=seed)
        state = env.reset()
        total = 0.0
        for step in range(env.horizon):
            action = agent.act(state, training=False) if policy == 'Q-learning' else fixed_timer_action(step)
            state, reward, _, _ = env.step(action)
            total += reward
        records.append({'policy': policy, 'seed': seed, 'total_reward': total, **env.metrics()})
    return records

def save_outputs(agent, rewards, evaluation):
    OUT.mkdir(exist_ok=True)
    np.save(OUT / 'q_table.npy', agent.q)
    with (OUT / 'training_rewards.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['episode', 'total_reward']); w.writerows(enumerate(rewards, 1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(rewards)+1), rewards, alpha=.22, label='Episode reward')
    window = min(50, len(rewards))
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window, len(rewards)+1), smooth, label=f'{window}-episode moving average')
    ax.set(xlabel='Training episode', ylabel='Total reward', title='Q-learning training rewards')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / 'training_rewards.png', dpi=160); plt.close(fig)
    with (OUT / 'evaluation.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(evaluation[0])); w.writeheader(); w.writerows(evaluation)
    policies = ['Fixed timer', 'Q-learning']
    means = [np.mean([r['mean_queue'] for r in evaluation if r['policy'] == p]) for p in policies]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(policies, means)
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set(ylabel='Mean vehicles waiting (lower is better)', title='Evaluation over identical random seeds')
    ax.set_ylim(0, max(means)*1.2 if max(means) > 0 else 1)
    fig.tight_layout(); fig.savefig(OUT / 'performance_comparison.png', dpi=160); plt.close(fig)
    return dict(zip(policies, means))

def main():
    parser = argparse.ArgumentParser(description='Q-learning traffic signal control')
    parser.add_argument('--episodes', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-runs', type=int, default=30)
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_runs < 1:
        parser.error('--episodes and --eval-runs must be >= 1')
    agent, rewards = train(args.episodes, args.seed)
    seeds = range(10000, 10000 + args.eval_runs)
    evaluation = evaluate(agent, seeds, 'Fixed timer') + evaluate(agent, seeds, 'Q-learning')
    means = save_outputs(agent, rewards, evaluation)
    print(f'Training complete: {args.episodes} episodes; evaluation: {args.eval_runs} scenarios per method')
    for label, value in means.items():
        print(f'{label}: mean queue = {value:.3f} vehicles')
    print(f'Outputs saved in: {OUT}')
    print('Note: this is a simplified simulation, not a real-world traffic safety controller.')

if __name__ == '__main__':
    main()
