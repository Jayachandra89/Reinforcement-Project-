# Intelligent Traffic Signal Optimisation Using Reinforcement Learning

### A Q-Learning-Based Adaptive Traffic Signal Control System

## 1. Project Overview

Traffic congestion is a major challenge in urban transportation systems. Conventional traffic signals generally operate using fixed-time intervals, which may not efficiently respond to changing traffic conditions.

This project implements an **Intelligent Traffic Signal Control System using Reinforcement Learning (RL)**. A Q-Learning agent interacts with a simulated traffic environment and learns an adaptive traffic signal control policy to minimise vehicle congestion and waiting time.

The performance of the trained agent is compared against a conventional fixed-timer traffic signal controller.

## 2. Project Objectives

- Design and implement a simulated traffic intersection.
- Develop an intelligent traffic control agent using Q-Learning.
- Implement a reward-based decision-making mechanism.
- Train the agent through repeated interactions with the environment.
- Compare adaptive and fixed-time traffic signal control methods.
- Visualise learning performance and traffic efficiency using graphs.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations and Q-table management |
| Matplotlib | Data visualisation |
| CSV | Storing training and evaluation results |
| Git & GitHub | Version control and project management |

The project uses a lightweight tabular Q-Learning implementation and does not require a dedicated GPU.

## 4. Reinforcement Learning Methodology

The proposed system follows the Reinforcement Learning framework, in which an agent learns through continuous interaction with its environment.

### RL Components

| Component | Description |
|---|---|
| Agent | Intelligent traffic signal controller |
| Environment | Simulated traffic intersection |
| State | Traffic queue conditions and current signal phase |
| Action | Maintain or switch the traffic signal |
| Reward | Feedback based on traffic congestion and switching costs |
| Policy | Learned Q-Learning decision-making strategy |

### Q-Learning Algorithm

Q-Learning is a model-free, off-policy Reinforcement Learning algorithm that estimates the expected long-term reward associated with performing an action in a given state.

The Q-value update equation is:

Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

Where:

- α = Learning rate
- γ = Discount factor
- r = Immediate reward
- s = Current state
- a = Selected action
- s' = Next state

An epsilon-greedy exploration strategy enables the agent to balance exploration and exploitation during training.

## 5. System Architecture

The project follows the workflow:

1. Initialise the simulated traffic environment.
2. Observe the current traffic state.
3. Select an action using an epsilon-greedy strategy.
4. Execute the selected traffic signal action.
5. Observe the resulting traffic conditions.
6. Calculate the reward.
7. Update the Q-table.
8. Repeat the process over multiple training episodes.
9. Evaluate the trained agent against a fixed-timer controller.

## 6. Project Directory Structure

```text
RL_Traffic_Signal_Project/
│
├── main.py
├── traffic_environment.py
├── q_learning_agent.py
├── fixed_timer.py
├── requirements.txt
├── .gitignore
│
├── tests/
│   └── test_project.py
│
├── results/
│   ├── evaluation.csv
│   ├── training_rewards.csv
│   ├── training_rewards.png
│   ├── performance_comparison.png
│   └── q_table.npy
│
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| main.py | Main execution and experiment workflow |
| traffic_environment.py | Traffic simulation environment |
| q_learning_agent.py | Q-Learning agent implementation |
| fixed_timer.py | Conventional fixed-time controller |
| requirements.txt | Required Python dependencies |
| test_project.py | Automated project tests |
| results/ | Training data, evaluation results and graphs |

## 7. Installation and Execution

### Prerequisites

- Python 3.10 or later
- pip package manager
- VS Code or another Python-compatible IDE

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/RL-Traffic-Signal-Q-Learning.git
```

Navigate to the project directory containing `main.py`.

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the Project

```bash
python main.py
```

The program performs agent training and evaluates the learned traffic control policy.

### Run Automated Tests

```bash
python -m unittest discover -s tests -v
```

## 8. Results and Performance Evaluation

The system compares a conventional fixed-time traffic controller with a trained Q-Learning controller.

The initial simulation produced the following results:

| Traffic Control Method | Average Vehicles Waiting |
|---|---:|
| Fixed-Timer Controller | 13.253 |
| Q-Learning Controller | 6.344 |

These results are obtained from the simulated environment and may vary depending on the experiment configuration and random seed.

### Training Reward Visualisation

![Training Rewards](results/training_rewards.png)

### Performance Comparison

![Performance Comparison](results/performance_comparison.png)

The graphs illustrate the learning behaviour of the RL agent and provide a comparative evaluation of the two traffic control strategies.

## 9. Advantages

- Demonstrates practical implementation of Reinforcement Learning.
- Adapts signal-control decisions according to simulated traffic conditions.
- Uses a lightweight Q-Learning approach.
- Provides measurable performance comparisons.
- Does not require expensive hardware or external datasets.
- Can be extended to more complex traffic management scenarios.

## 10. Limitations

- The project uses a simplified traffic simulation rather than real-world traffic data.
- Tabular Q-Learning may not scale efficiently to large state spaces.
- Performance depends on reward design and training parameters.
- Real-world deployment would require additional safety constraints, traffic modelling and validation.

## 11. Future Scope

Potential improvements include:

- Implementing Deep Q-Networks (DQN).
- Integrating live traffic sensor data.
- Introducing multiple traffic intersections.
- Using computer vision for vehicle detection.
- Implementing emergency vehicle prioritisation.
- Integrating the simulation with SUMO.
- Developing a real-time graphical dashboard.

## 12. Conclusion

This project demonstrates how Reinforcement Learning can be applied to adaptive traffic signal management.

By implementing a Q-Learning agent and comparing its performance against a conventional fixed-time controller, the system explores the potential of reward-based learning to improve traffic management decisions.

The project provides a foundation for further research into intelligent transportation systems and AI-based traffic optimisation.

---

**Project Category:** Artificial Intelligence / Reinforcement Learning  
**Algorithm:** Q-Learning  
**Language:** Python  
**Purpose:** Academic Mini Project – Internal Assessment
