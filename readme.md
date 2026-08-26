# Unity ML-Agents Quadruped Training Environment 🐕

A custom Deep Reinforcement Learning environment built in Unity using the ML-Agents toolkit. This project trains a 4-legged robot (quadruped) to balance, walk, and navigate towards a dynamic target using Proximal Policy Optimization (PPO).

## 🎮 Project Overview

This project simulates a physics-based quadruped robot using Unity's `ArticulationBody` system. Rather than using traditional spring-based Rigidbodies, the robot is built with `xDrive` to simulate stiff, precise, and realistic robotic servo motors. 

The AI agent learns to coordinate all 8 distinct leg joints to propel itself forward without falling over, jumping, or crashing into walls. To achieve high throughput during training, the environment is scaled to **100 parallel agents** and optimized for headless background training.

## 🛠️ Tech Stack

* **Engine:** Unity (C#)
* **ML Framework:** Unity ML-Agents Toolkit
* **Algorithm:** Proximal Policy Optimization (PPO) via PyTorch
* **Physics:** Unity ArticulationBody System

## 🧠 Agent Architecture (`QuadrupedAgent.cs`)

### Observation Space (Size: 23 Continuous)
The agent observes its physical state and environment 60 times a second. It requires momentum and gravity data to successfully learn how to balance:

1. `Vector3` Direction to target (normalized) [3]
2. `Vector3` Forward direction of the main body [3]
3. `Vector3` Linear Velocity (momentum) [3]
4. `Vector3` Angular Velocity (spin/fall detection) [3]
5. `Vector3` Up Vector (gravity/balance detection) [3]
6. `float` Joint positions for all 8 legs [8]

### Action Space (Size: 8 Continuous)
The neural network outputs 8 continuous values between `-1.0` and `1.0`. These are normalized and mapped to the physical `lowerLimit` and `upperLimit` of each leg's `xDrive` target.

### Reward Shaping
The reward function is meticulously shaped to encourage smooth, efficient locomotion and discourage "reward hacking" (like jumping or locking joints):

* **+ Velocity Incentive:** Micro-rewards every step based on the dot product of the dog's velocity and the direction to the target (forces it to move forward).
* **- Energy Penalty:** Squares the motor outputs and subtracts them as a penalty (forces smooth, low-energy walking over violent thrashing).
* **- Anti-Air Penalty:** Penalizes the agent heavily if it jumps too high, forcing it to walk instead of acting like a kangaroo.
* **+ Success Bonus (+1.0):** Reaching within 1.5m of the dynamic target.
* **- Failure Penalty (-1.0):** Face-planting (belly hits the floor) or colliding with walls triggers an instant episode reset.

## 🚀 How to Train (Headless Mode)

To maximize hardware performance and bypass the Unity Editor's rendering overhead, training is done via a standalone executable.

**1. Build the Environment:**
Build the Unity Scene containing the 100 agents as a standalone Windows executable to your `Builds/` folder (e.g., `My project (1).exe`).

**2. Run ML-Agents (Terminal):**
Open your terminal in the project root and run the following command to begin headless training:

```bash
mlagents-learn config/quadruped.yaml --env="Builds/DogTraining_New/My project (1)" --run-id=Dog_Run_01 --no-graphics
