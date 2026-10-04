# Intelligent Traffic Signal Optimisation Using Q-Learning

A small, reproducible reinforcement learning mini-project for a **simulated** two-phase road intersection. Tabular Q-learning learns when to keep the current green phase or switch it, and is compared against a fixed-timer policy.

## Requirements
- Python 3.10+
- No GPU or downloaded dataset required

## Quick start
```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --episodes 1500 --eval-runs 30
```
For a fast smoke test: `python main.py --episodes 10 --eval-runs 3`.

## Reinforcement learning design
- **Agent:** signal controller.
- **Environment:** two queues: North/South and East/West; time advances in 10-second slots.
- **State:** (discretised NS queue, discretised EW queue, current green phase). Each queue has 6 buckets.
- **Actions:** 0 = keep current phase; 1 = switch phase.
- **Arrivals:** sampled from independent Poisson distributions (2.0 and 2.5 vehicles per slot). Service capacity is 5 vehicles per slot on the green phase.
- **Reward:** negative remaining total queue minus 2 for a phase switch.
- **Update:** Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]; at terminal step, no next-state value is added.
- **Training:** epsilon-greedy exploration, epsilon decay, one 120-step episode per seed.
- **Comparator:** fixed timer switching every 3 slots. Both policies are evaluated with the **same set of random seeds**, independently from training. This is an illustrative educational model, not a road-ready safety solution.

## Output files
Running `main.py` creates/updates:
- `results/training_rewards.csv` and `results/training_rewards.png`
- `results/evaluation.csv` and `results/performance_comparison.png`
- `results/q_table.npy` (trained Q values)

**Metric note:** `mean_queue` is the time average of vehicles still waiting after departures each slot. `waiting_slots_per_departure` is a proxy based on cumulative queued vehicle-slots divided by the number of departures; it is **not** an exact measured individual vehicle delay. Compare policies on the same seeds; outcomes depend on modelling assumptions and hyperparameters. Do not claim a guaranteed RL improvement if your run does not demonstrate one.

## Source layout
- `traffic_environment.py`: traffic dynamics, observations, rewards, metrics.
- `q_learning_agent.py`: tabular epsilon-greedy Q-learning.
- `fixed_timer.py`: baseline policy.
- `main.py`: command-line training, independent evaluation, chart generation.
- `tests/test_project.py`: quick standard-library tests.

## Tests
```bash
python -m unittest discover -s tests -v
```

## GitHub upload (command line)
1. Create an **empty public or private GitHub repository** named `RL-Traffic-Signal-Q-Learning` (do not initialise it with a README).
2. From this folder run:
```bash
git init
git add .
git commit -m "Initial Q-learning traffic signal project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/RL-Traffic-Signal-Q-Learning.git
git push -u origin main
```
Replace `YOUR_USERNAME` with your actual GitHub username. GitHub may ask you to sign in via your browser or credential manager. Never commit passwords or access tokens.

## Limitations
The environment uses only two traffic phases, simplified Poisson arrivals, no amber/all-red clearance interval, no pedestrian phases, no emergency vehicles, and discretised queues. Real-world deployment would require safety interlocks, calibration, detailed traffic simulation, and comprehensive testing.
