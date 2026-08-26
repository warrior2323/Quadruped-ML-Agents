# 🐕 Quadruped Locomotion via Deep Reinforcement Learning

A four-legged robotic dog that **teaches itself to walk** using Deep Reinforcement Learning inside a Unity 6 physics simulation. No hand-coded animations, no motion capture — just a neural network, a reward signal, and millions of trial-and-error episodes.

<p align="center">
  <b>Engine:</b> Unity 6 (6000.2.8f1) &nbsp;|&nbsp;
  <b>AI Framework:</b> Unity ML-Agents 2.0.2 &nbsp;|&nbsp;
  <b>Algorithm:</b> Proximal Policy Optimization (PPO) &nbsp;|&nbsp;
  <b>Backend:</b> PyTorch
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [What Did I Actually Do?](#what-did-i-actually-do)
- [Core AI Concepts](#core-ai-concepts)
- [Architecture & How It Works](#architecture--how-it-works)
- [Prerequisites & Installation](#prerequisites--installation)
- [Training the AI from Scratch](#training-the-ai-from-scratch)
  - [Option A — Train with CPU](#option-a--train-with-cpu)
  - [Option B — Train with GPU (CUDA)](#option-b--train-with-gpu-cuda)
  - [Train WITH Graphics (Watch it Learn in Unity)](#train-with-graphics-watch-it-learn-in-unity)
  - [Train WITHOUT Graphics (Headless / Terminal Only)](#train-without-graphics-headless--terminal-only)
- [Testing the Pre-Trained Model](#testing-the-pre-trained-model)
- [Viewing Training Graphs (TensorBoard)](#viewing-training-graphs-tensorboard)
- [Training Configuration Explained](#training-configuration-explained)
- [Observation & Reward System Deep Dive](#observation--reward-system-deep-dive)
- [Project Structure](#project-structure)
- [Training Runs & Results Summary](#training-runs--results-summary)
- [Mistakes & Lessons Learned](#mistakes--lessons-learned)
- [Troubleshooting / FAQ](#troubleshooting--faq)

---

## Project Overview

This project trains a **quadruped (four-legged) robot dog** to walk towards a randomly-placed target box inside a Unity 3D environment. The dog has **8 motorized joints** (2 per leg — a hip and a knee), and a neural network controls all 8 motors simultaneously every physics frame.

The dog is never told *how* to walk. Instead, we define:
- **Rewards** → "You moved towards the target? Good boy. +points."
- **Penalties** → "You fell on your face? Bad dog. -points."

Over millions of episodes, the AI discovers walking gaits entirely on its own through Proximal Policy Optimization (PPO).

---

## What Did I Actually Do?

This project is an **end-to-end Deep Reinforcement Learning pipeline** built from scratch. Here's what was involved:

1. **Designed the 3D environment** in Unity 6 — a flat ground plane, boundary walls, and a target box that spawns at random positions within a 15-meter radius.

2. **Built a physically-accurate quadruped robot** using Unity's `ArticulationBody` physics system (not standard Rigidbodies). ArticulationBodies simulate stiff, precise servo motors via `xDrive`, which is critical for robotic locomotion — standard spring-based joints are too floppy and rubbery for stable learning.

3. **Wrote the AI agent script** (`QuadrupedAgent.cs`) that bridges the Unity physics engine with the ML-Agents neural network. This script defines:
   - **23 sensory observations** the brain receives each frame (direction to target, joint angles, velocity, angular velocity, orientation).
   - **8 continuous motor outputs** the brain controls (one per leg joint).
   - A **dense reward shaping** system with velocity incentives, energy penalties, height penalties, and terminal rewards/punishments.

4. **Engineered the training configuration** (`config/quadruped.yaml`) — tuned PPO hyperparameters including batch size, buffer size, learning rate scheduling, network architecture (3 hidden layers × 256 neurons), and a 50-million-step training budget.

5. **Ran multiple training experiments** across CPU and GPU, with and without graphics, iterated on reward functions, debugged physics instabilities, and produced trained `.onnx` neural network models that can run inference in real-time inside Unity.

6. **Built a standalone headless executable** (`Builds/DogTraining_New/`) for maximum training throughput — running the simulation without rendering graphics to achieve significantly higher steps-per-second.

---

## Core AI Concepts

### Reinforcement Learning (RL)

Reinforcement Learning is a type of machine learning where an **agent** learns to make decisions by performing **actions** in an **environment** to maximize a cumulative **reward**. Instead of programming "move leg A by 20 degrees, then leg B," we simply reward the dog for moving forward and penalize it for falling. Through millions of trial-and-error attempts, the agent figures out the mechanics itself.

### Deep Reinforcement Learning (Deep RL)

Standard RL works for simple games where every possible state can be memorized in a table. However, our 3D physics environment is **continuous** — joint angles, velocity, and rotation have infinite possible values. Deep RL solves this by replacing the lookup table with a **Deep Neural Network**. The network takes in the current physical state as numbers, processes them through hidden layers, and outputs the best motor forces to apply.

### Proximal Policy Optimization (PPO)

PPO is the specific algorithm acting as the dog's "brain." Developed by OpenAI, PPO balances speed and stability. When the AI discovers a good move, older algorithms would aggressively rewrite their neural networks to repeat it, often causing **catastrophic forgetting** (forgetting how to balance just to lunge forward). PPO uses mathematical **clipping** (ε = 0.2) to ensure the network only updates its policy by a small, safe percentage at a time — guaranteeing smooth, steady learning.

---

## Architecture & How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (per step)                   │
│                                                              │
│  ┌──────────────┐    23 floats     ┌──────────────────────┐  │
│  │ Unity Engine  │ ──────────────→ │   Neural Network     │  │
│  │ (Environment) │                 │   (PPO / PyTorch)    │  │
│  │               │ ←────────────── │                      │  │
│  │  • Physics    │   8 motor       │  3 Hidden Layers     │  │
│  │  • Collisions │   commands      │  256 Neurons Each    │  │
│  │  • Rewards    │                 │                      │  │
│  └──────────────┘                  └──────────────────────┘  │
│         │                                    │               │
│         └──── Reward Signal (+/-) ───────────┘               │
└──────────────────────────────────────────────────────────────┘

Observation Space (23 floats):
├── Direction to target (normalized)     → 3 values
├── Body forward direction               → 3 values
├── Linear velocity                      → 3 values
├── Angular velocity                     → 3 values
├── Body up direction (gravity sense)    → 3 values
└── Joint angles (8 legs)                → 8 values

Action Space (8 continuous values):
└── Motor target angle for each joint    → [-1.0, +1.0] mapped to joint limits
```

---

## Prerequisites & Installation

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Unity Hub** | Latest | Project management & editor installation |
| **Unity Editor** | `6000.2.8f1` (Unity 6) | Open and run the project |
| **Python** | `3.10.x` (tested with 3.10.11) | ML-Agents training backend |
| **pip** | Latest | Python package manager |
| **Git** | Latest | Clone this repository |
| **CUDA Toolkit** | 12.8+ *(GPU only)* | GPU-accelerated training |
| **cuDNN** | Compatible version *(GPU only)* | Deep learning GPU primitives |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/warrior2323/Quadruped-ML-Agents.git
cd Quadruped-ML-Agents
```

### Step 2 — Set Up Python Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install ML-Agents & Dependencies

```bash
# Install the ML-Agents Python package
pip install mlagents==1.2.0.dev0

# This automatically installs PyTorch (CPU version).
# For GPU training, see the GPU section below.

# Verify installation
mlagents-learn --help
```

### Step 4 — Open the Unity Project

1. Open **Unity Hub**
2. Click **"Add" → "Add project from disk"**
3. Navigate to the cloned folder and select it
4. Unity Hub will detect the required editor version (`6000.2.8f1`). Install it if prompted.
5. Open the project. Unity will import all assets.
6. Open the scene: `Assets/Scenes/SampleScene.unity`

---

## Training the AI from Scratch

> **Important:** Always make sure your Python virtual environment is activated before running any training commands.

### Option A — Train with CPU

CPU training works out of the box with the default `mlagents` installation. It is slower but requires no special hardware.

**Step 1:** Make sure PyTorch CPU is installed (it's the default):
```bash
pip install mlagents==1.2.0.dev0
# This installs torch with +cpu automatically
```

**Step 2:** Start training:
```bash
mlagents-learn config/quadruped.yaml --run-id=MyFirstRun_CPU --force
```

**Step 3:** When you see the Unity logo in the terminal and the message:
```
[INFO] Listening on port 5005. Start training by pressing the Play button in the Unity Editor.
```
Go to the Unity Editor and press the **▶ Play** button.

Training will begin. You'll see stats printed every 50,000 steps:
```
[INFO] DogBrain. Step: 50000. Time Elapsed: 120.456 s. Mean Reward: -8.234. Std of Reward: 3.456.
```

> **Note:** The CPU run (`Dog_Run_01`) used PyTorch `2.12.0+cpu`. Expect ~500K steps to take ~20 minutes.

---

### Option B — Train with GPU (CUDA)

GPU training is **significantly faster** for neural network forward/backward passes. This project was tested with **CUDA 12.8**.

**Step 1:** Install CUDA-enabled PyTorch (replace the CPU version):
```bash
# Uninstall CPU PyTorch first
pip uninstall torch torchvision torchaudio -y

# Install CUDA 12.8 version (adjust cu128 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Step 2:** Verify GPU is detected:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```
Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX XXXX
```

**Step 3:** Start training (same command — ML-Agents auto-detects the GPU):
```bash
mlagents-learn config/quadruped.yaml --run-id=MyFirstRun_GPU --force
```

**Step 4:** Press **▶ Play** in Unity Editor (or use the headless build — see below).

> **Note:** The GPU runs (`Dog_Run_GPU_01`, `Dog_Run_GPU_02`) used PyTorch `2.11.0+cu128` and achieved 24+ million steps.

---

### Train WITH Graphics (Watch it Learn in Unity)

This method lets you **see the dog learning in real-time** inside the Unity Editor. It's great for debugging and visualization, but slower because Unity has to render every frame.

```bash
# 1. Start the training server
mlagents-learn config/quadruped.yaml --run-id=VisualRun --force

# 2. Wait for "Listening on port 5005..."

# 3. Press ▶ Play in the Unity Editor

# 4. Watch the dog train live! You'll see it stumbling, falling, and gradually learning to walk.
```

**Tips for in-editor training:**
- The `time_scale` in the config is set to `20`, meaning the simulation runs 20× faster than real-time even with graphics.
- You can adjust the camera in the Scene/Game view to follow the dog.
- Training data is logged to `results/<run-id>/` in real-time.

---

### Train WITHOUT Graphics (Headless / Terminal Only)

This is the **fastest training method**. You use a pre-compiled standalone build of the environment and run everything from the terminal — no Unity Editor required. The simulation runs with a Null graphics device, maximizing step throughput.

**Step 1:** Use the pre-built executable (already included at `Builds/DogTraining_New/`):

```bash
mlagents-learn config/quadruped.yaml --run-id=HeadlessRun --force --env="Builds/DogTraining_New/My project (1).exe" --no-graphics
```

**Or, if you want to build your own executable:**

1. In Unity Editor: `File → Build Settings`
2. Select **Windows, Mac, Linux** as the platform
3. Click **Build** and choose an output directory (e.g., `Builds/MyBuild/`)
4. Run:
```bash
mlagents-learn config/quadruped.yaml --run-id=HeadlessRun --force --env="Builds/MyBuild/YourExeName.exe" --no-graphics
```

**Flags explained:**

| Flag | Description |
|------|-------------|
| `--env="path/to/build.exe"` | Path to the standalone Unity build (bypasses the Editor entirely) |
| `--no-graphics` | Forces a Null GPU device — no window, no rendering, maximum speed |
| `--force` | Overwrites any previous run with the same `--run-id` |
| `--run-id=<name>` | A unique name for this training run (results saved under `results/<name>/`) |

> **Note:** Shader warnings in the Player log (e.g., "Shader not supported on this GPU") are **expected and harmless** when using `--no-graphics`. The Null renderer simply can't execute GPU shaders, which is fine because we don't need them.

---

### Resuming a Training Run

If training is interrupted (power loss, manual stop, etc.), you can resume from the last checkpoint:

```bash
mlagents-learn config/quadruped.yaml --run-id=Dog_Run_GPU_01 --resume
```

This will:
- Load the latest `checkpoint.pt` from `results/Dog_Run_GPU_01/DogBrain/`
- Continue training from where it left off
- Append new TensorBoard events to the existing log

---

## Testing the Pre-Trained Model

A pre-trained neural network model is included at:
```
Assets/Brains/DogBrain-12890424.onnx
```

This `.onnx` (Open Neural Network Exchange) file was exported from the `Dog_Run_GPU_01` training run at step **12,890,424** and can be used for real-time inference inside Unity.

### Steps to Test

1. **Open the Unity project** and load `Assets/Scenes/SampleScene.unity`

2. **Assign the trained brain to the agent:**
   - Select the **Dog** GameObject in the Hierarchy
   - Find the `Behavior Parameters` component in the Inspector
   - Set **Behavior Type** to `Inference Only`
   - Drag `Assets/Brains/DogBrain-12890424.onnx` into the **Model** field
   - Set **Inference Device** to `CPU` (or `GPU` if you have a compatible GPU)

3. **Press ▶ Play** — The dog will use the trained neural network to walk towards the target box in real-time!

> **Tip:** If the dog doesn't move or behaves erratically, make sure the `Behavior Name` in the `Behavior Parameters` component is set to `DogBrain` — this must match the behavior name in the training config.

---

## Viewing Training Graphs (TensorBoard)

Training metrics are logged as TensorBoard event files inside each run's directory. You can visualize learning curves, reward progression, policy loss, and more.

### Launch TensorBoard

```bash
# View a specific run
tensorboard --logdir=results/Dog_Run_GPU_01

# View ALL runs side-by-side for comparison
tensorboard --logdir=results

# Specify a custom port if 6006 is in use
tensorboard --logdir=results --port=6007
```

Then open your browser and navigate to:
```
http://localhost:6006
```

### Key Metrics to Monitor

| Metric | What It Means | Healthy Trend |
|--------|--------------|---------------|
| **Environment/Cumulative Reward** | Average total reward per episode | 📈 Should increase over time |
| **Environment/Episode Length** | Average number of steps per episode | 📈 Longer = dog survives longer |
| **Losses/Policy Loss** | How much the policy is changing | 📉 Should decrease and stabilize |
| **Losses/Value Loss** | Error in the value function estimate | 📉 Should decrease over time |
| **Policy/Entropy** | Randomness in the agent's actions | 📉 Decreases as the agent becomes more confident |
| **Policy/Learning Rate** | Current learning rate (linear schedule) | 📉 Linearly decays from 0.0003 → 0 |
| **Policy/Extrinsic Value Estimate** | The agent's prediction of future reward | 📈 Should increase as the agent learns |

### Example: Comparing CPU vs GPU Runs

```bash
# This will show all three runs overlaid on the same graphs
tensorboard --logdir=results
```

You'll see runs named:
- `Dog_Run_01` — CPU training run (~500K steps, PyTorch 2.12.0+cpu)
- `Dog_Run_GPU_01` — Primary GPU run (~24.2M steps, PyTorch 2.11.0+cu128)
- `Dog_Run_GPU_02` — Secondary GPU run (~158K steps, PyTorch 2.11.0+cu128)

---

## Training Configuration Explained

The training hyperparameters are defined in [`config/quadruped.yaml`](config/quadruped.yaml):

```yaml
behaviors:
  DogBrain:                          # Must match the Behavior Name in Unity Inspector
    trainer_type: ppo                # Proximal Policy Optimization algorithm
    hyperparameters:
      batch_size: 2048               # Samples per gradient update
      buffer_size: 40960             # Total experience buffer before update
      learning_rate: 0.0003          # Initial learning rate
      beta: 0.005                    # Entropy regularization (exploration bonus)
      epsilon: 0.2                   # PPO clipping range (prevents large policy jumps)
      lambd: 0.95                    # GAE lambda (bias-variance tradeoff)
      num_epoch: 3                   # Passes over buffer per update
      learning_rate_schedule: linear # Decays LR to 0 over training
    network_settings:
      normalize: true                # Normalizes observations for stable training
      hidden_units: 256              # Neurons per hidden layer
      num_layers: 3                  # Number of hidden layers in the network
    reward_signals:
      extrinsic:
        gamma: 0.99                  # Discount factor (values future rewards highly)
        strength: 1.0                # Multiplier for the reward signal
    max_steps: 50000000              # Total training budget (50 million steps)
    time_horizon: 1000               # Steps collected before computing advantages
    summary_freq: 50000              # Log stats to TensorBoard every 50K steps
```

### Key Hyperparameter Decisions

| Parameter | Value | Why |
|-----------|-------|-----|
| `hidden_units: 256` | 256 neurons | Large enough for 23 inputs → 8 outputs mapping of continuous locomotion |
| `num_layers: 3` | 3 layers deep | Deeper network captures complex joint coordination patterns |
| `buffer_size: 40960` | 40K | ~20× the batch size, providing diverse training experience |
| `time_horizon: 1000` | 1000 steps | Long horizon so the dog can experience full walking sequences before credit assignment |
| `gamma: 0.99` | 0.99 | The agent values rewards far into the future (important for locomotion goals) |
| `normalize: true` | Enabled | Critical — observations span different scales (angles vs velocities vs directions) |

---

## Observation & Reward System Deep Dive

### Observation Space (23 values)

| Observation | Count | Purpose |
|-------------|-------|---------|
| `directionToTarget.normalized` | 3 | Where the target is relative to the dog |
| `mainBody.transform.forward` | 3 | Which way the dog is currently facing |
| `mainBody.linearVelocity` | 3 | How fast the dog is moving (and in what direction) |
| `mainBody.angularVelocity` | 3 | Is the dog spinning or tumbling? |
| `mainBody.transform.up` | 3 | Which way is "up" for the dog (gravity sense) |
| `leg.jointPosition[0]` × 8 | 8 | Current angle of each of the 8 leg joints |

> **Why this was chosen:** Early in development, the dog was only fed joint angles and target position. It failed to learn because the brain couldn't feel momentum or gravity. Adding velocity and the "Up" vector allowed the AI to *feel* itself falling — which is the prerequisite to learning how to catch itself.

### Reward System

| Signal | Value | When | Purpose |
|--------|-------|------|---------|
| **Velocity Incentive** | `+forwardSpeed × 0.1` | Every step | Rewards speed towards the target |
| **Energy Penalty** | `-Σ(action²) × 0.005` | Every step | Punishes wild, wasteful movements |
| **Height Penalty** | `-0.05` | When `y > 2.25` | Prevents jumping/bouncing exploits |
| **Goal Reached** | `+1.0` (terminal) | Distance < 1.5m | Massive bonus for success |
| **Fell Down** | `-1.0` (terminal) | `y < 1.2` | Massive penalty for collapsing |
| **Hit Wall** | `-1.0` (terminal) | Collision with wall | Massive penalty for crashing |

The velocity incentive uses `Vector3.Dot(velocity, directionToTarget)` — this only rewards speed that is pointed *directly at the target*, not sideways or backwards movement.

The energy penalty squares each motor output: an action of `0.1` costs almost nothing (`0.01 × 0.005`), but an extreme action of `1.0` costs significantly more (`1.0 × 0.005`). This teaches the AI smooth, efficient gaits.

---

## Project Structure

```
Quadruped-ML-Agents/
│
├── Assets/
│   ├── Brains/
│   │   └── DogBrain-12890424.onnx      # ← Pre-trained neural network model
│   ├── ML-Agents/
│   │   └── Timers/                      # Runtime timer data from ML-Agents
│   ├── Scenes/
│   │   └── SampleScene.unity            # ← Main Unity scene with the dog & environment
│   ├── QuadrupedAgent.cs                # ← The core AI agent script
│   └── InputSystem_Actions.inputactions # Input system configuration
│
├── Builds/
│   └── DogTraining_New/                 # ← Pre-compiled headless training build
│       ├── My project (1).exe           #    Standalone executable
│       └── ...                          #    Supporting DLLs and data
│
├── config/
│   └── quadruped.yaml                   # ← PPO training hyperparameters
│
├── results/                             # ← All training run outputs
│   ├── Dog_Run_01/                      #    CPU training run
│   │   ├── DogBrain/                    #    Checkpoints (.pt) & TF events
│   │   ├── run_logs/                    #    Timers and training status
│   │   └── configuration.yaml           #    Frozen config snapshot
│   ├── Dog_Run_GPU_01/                  #    Primary GPU training run
│   │   ├── DogBrain/                    #    268 checkpoint files, ONNX models
│   │   │   └── DogBrain-12890424.onnx   #    Best model (used in Brains/)
│   │   ├── DogBrain.onnx                #    Final exported model
│   │   └── run_logs/                    #    Player logs, timers, status
│   └── Dog_Run_GPU_02/                  #    Secondary GPU run (short)
│
├── Packages/
│   ├── manifest.json                    # Unity package dependencies
│   └── packages-lock.json               # Locked dependency versions
│
├── ProjectSettings/                     # Unity project settings
├── Doc.docx                             # Original project documentation
├── .gitignore                           # Git ignore rules
└── README.md                            # ← You are here
```

---

## Training Runs & Results Summary

| Run | Device | PyTorch | Total Steps | Mean Reward (Final) | Status |
|-----|--------|---------|-------------|---------------------|--------|
| `Dog_Run_01` | CPU | `2.12.0+cpu` | ~500K | -5.36 (early stage) | Completed |
| `Dog_Run_GPU_01` | GPU (CUDA 12.8) | `2.11.0+cu128` | ~24.2M | +48.1 to +50.4 | ✅ Best Run |
| `Dog_Run_GPU_02` | GPU (CUDA 12.8) | `2.11.0+cu128` | ~158K | -92.5 (just started) | Early termination |

### Key Observations

- **CPU Run (`Dog_Run_01`)**: Ran for ~500K steps in ~1,222 seconds (~20 min). The cumulative reward was still deeply negative (-5.36), meaning the dog was still in the early "falling over" phase. CPU training is significantly slower — it's mainly useful for verifying the setup works.

- **GPU Run (`Dog_Run_GPU_01`)**: This was the primary training run, reaching **24.2 million steps** with a final cumulative reward of **~48–50**. The dog successfully learned to walk toward the target box. The model exported from step 12,890,424 is stored in `Assets/Brains/` for inference.

- **GPU Run (`Dog_Run_GPU_02`)**: A short experimental run that was terminated early at 158K steps (reward: -92.5). Likely a hyperparameter experiment or restart.

---

## Mistakes & Lessons Learned

These hard-won lessons shaped the final working system:

### 1. The "Vertical Spawn" Physics Bug 🚀
**Problem:** When the dog died and reset, Unity's physics engine retained the twisted joint angles from the crash. Because the script originally didn't save/restore `startingRotation`, the tangled legs caused massive physics collisions on frame 1, launching the dogs vertically into the air.

**Fix:** Cache both `startingPosition` and `startingRotation` during `Initialize()`, then apply both via `TeleportRoot(startingPosition, startingRotation)` on every `OnEpisodeBegin()`.

### 2. The Instant-Death Loop 💀
**Problem:** The death penalty was too strict — triggering if the dog's Y-height dipped below `0.5f`. Since the dog spawns at roughly `0.8f`, simply bending its knees to take a step would kill it instantly. The AI learned that "moving = death" and froze in mid-air.

**Fix:** Relaxed the death threshold to `y < 1.2f` and removed the overly strict rotation-flip check.

### 3. The OS Build Lockout 🔒
**Problem:** During iteration, builds started failing instantly with an "Unknown" error. This was caused by Windows locking the `UnityPlayer.dll` because the previous training executable was still silently running in the background.

**Fix:** Always terminate previous training processes before rebuilding. Push builds to fresh, empty directories (e.g., `Builds/DogTraining_New/`).

### 4. Missing Sensory Inputs 🧠
**Problem:** Early versions only fed the neural network joint angles and target position. The dog couldn't learn to walk because it had no sense of momentum or gravity.

**Fix:** Added `linearVelocity`, `angularVelocity`, and `transform.up` to the observation space — giving the dog the ability to "feel" falling and spinning.

---

## Troubleshooting / FAQ

### "Listening on port 5005" but Unity won't connect
- Make sure you press **▶ Play** in the Unity Editor *after* the terminal shows the listening message.
- If using a build with `--env`, make sure the path to the `.exe` is correct.
- Check that no other `mlagents-learn` process is already occupying port 5005:
  ```bash
  # Kill any existing training processes
  taskkill /f /im "My project (1).exe" 2>$null
  ```

### "No Behavior found" or "No Agent found"
- Ensure the `Behavior Name` field in the Unity Inspector's `Behavior Parameters` component is set to exactly `DogBrain` (case-sensitive).
- This must match the key in `config/quadruped.yaml` under `behaviors:`.

### CUDA / GPU not detected
```bash
# Check CUDA installation
nvcc --version

# Check PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with the correct CUDA version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Shader errors in headless mode
Errors like `"Shader Hidden/Universal Render Pipeline/... not supported on this GPU"` are **completely normal** when using `--no-graphics`. The Null renderer cannot execute GPU shaders, but this does not affect training at all.

### "Run with the same ID already exists"
Use `--force` to overwrite or `--resume` to continue:
```bash
# Overwrite (delete old run)
mlagents-learn config/quadruped.yaml --run-id=MyRun --force

# Resume from checkpoint
mlagents-learn config/quadruped.yaml --run-id=MyRun --resume
```

### Dog doesn't move during inference
- Make sure `Behavior Type` is set to `Inference Only` (not `Default` or `Heuristic`).
- Ensure the `.onnx` model file is dragged into the `Model` field.
- Check the Unity Console for any errors.

### How to increase training speed
1. **Use headless builds** with `--env` and `--no-graphics` (biggest improvement)
2. **Use GPU** with CUDA-enabled PyTorch
3. **Increase `time_scale`** in the config (default: 20, try 40–100)
4. **Run multiple environments** with `--num-envs=N` (requires N copies of the build)

---

## License

This project is for educational and research purposes.

---

<p align="center">
  Built with ❤️ using Unity 6 + ML-Agents + PyTorch
</p>
