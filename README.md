# DG²-RL: Disturbance-Guided Grasp Decision Reinforcement Learning

## Overview

DG²-RL (Disturbance-Guided Grasp Decision Reinforcement Learning) is a hierarchical robotic grasping framework that combines visual grasp prediction, disturbance-aware state estimation, and PPO-based reinforcement learning for robust grasp execution under uncertainty.

The system is designed for aerial and mobile robotic manipulation scenarios where environmental disturbances, target motion, and perception uncertainty can significantly degrade grasp reliability.

The deployment platform consists of:

* Raspberry Pi
* Pi Camera
* Arduino Nano
* L298N Motor Driver
* N20 DC Motor / Linear Actuator
* PPO-based Meta Decision Controller

---

# System Architecture

```text
Pi Camera
    │
    ▼
Module 1
Uncertainty-Aware Grasp Prediction
    │
    ▼
Top-K Grasp Proposals
    │
    ▼
Module 2
Disturbance-Aware State Estimation
    │
    ▼
10D State Vector
    │
    ▼
Module 3
PPO Meta Decision Policy
    │
    ▼
Action Selection

WAIT
TRACK
REPOSITION
EXECUTE
ABORT

    │
    ▼
Arduino Nano
    │
    ▼
L298N Motor Driver
    │
    ▼
Motor / Linear Actuator
```

---

# Project Structure

```text
DG2RL/

│
├── main.py
│
├── config/
│   └── config.py
│
├── models/
│   ├── grasp_net.py
│   └── ppo_network.py
│
├── perception/
│   ├── module1_inference.py
│   ├── grasp_extraction.py
│   ├── objectness.py
│   └── state_builder.py
│
├── motion/
│   ├── disturbance.py
│   ├── optical_flow.py
│   └── temporal_features.py
│
├── rl/
│   ├── environment.py
│   ├── rollout_buffer.py
│   └── ppo_agent.py
│
├── hardware/
│   ├── arduino_controller.py
│   └── execution.py
│
├── visualization/
│   ├── hud.py
│   ├── debug.py
│   └── metrics.py
│
├── checkpoints/
│   ├── dg2_graspnet_model.pth
│   └── dg2rl_ppo_policy.pth
│
└── README.md
```

---

# Module 1: Multi-Grasp Prediction

## Objective

Generate multiple grasp proposals from RGB-D observations.

## Inputs

* RGB image
* Depth image

## Network Outputs

* Grasp Quality Map (Q)
* Cosine Map
* Sine Map
* Grasp Width Map

## Output

Top-K grasp candidates:

```python
{
    "x": x,
    "y": y,
    "angle": theta,
    "width": width,
    "score": confidence
}
```

---

# Module 2: Disturbance-Aware State Estimation

## Objective

Convert visual observations into a compact RL state representation.

## Features

### Grasp Features

* Maximum confidence
* Confidence uncertainty

### Disturbance Features

* Disturbance amplitude
* Disturbance frequency

### Motion Features

* Optical flow velocity
* Positional error

### Temporal Features

* Stability score
* Confidence change
* Error rate

---

## Final State Vector

```text
State =

[
 q_max,
 sigma_max,

 A_t,
 f_t,

 e_p,
 ep_dot,

 S_t,
 delta_q,

 v_d,
 omega_sin
]
```

State Dimension = 10

---

# Module 3: PPO Meta-Decision Policy

## Action Space

```text
0 → WAIT

1 → TRACK

2 → REPOSITION

3 → EXECUTE

4 → ABORT
```

---

## PPO Network

### Shared Encoder

```text
10
 ↓
128
 ↓
128
```

### Policy Head

```text
128 → 5
```

### Value Head

```text
128 → 1
```

---

# Reward Design

## WAIT

Encouraged during:

* High disturbance
* High uncertainty
* Low stability

---

## TRACK

Encouraged during:

* Small positional errors

---

## REPOSITION

Encouraged during:

* Large positional errors

---

## EXECUTE

Encouraged during:

* High confidence
* Low uncertainty
* Low disturbance
* High stability

---

## ABORT

Encouraged during:

* Catastrophic conditions
* Unsafe grasp situations

---

# Hardware Integration

## Raspberry Pi

Responsibilities:

* Camera acquisition
* State estimation
* PPO inference

---

## Arduino Nano

Responsibilities:

* Motor control
* Actuator triggering

Communication:

```text
Serial 9600 baud
```

Commands:

```text
'1' → EXECUTE

'0' → STOP
```

---

# L298N Connections

## Arduino

```text
D8 → IN1

D9 → IN2
```

## Motor

```text
OUT1 → Motor Terminal 1

OUT2 → Motor Terminal 2
```

## Power

```text
External Supply → L298N

Common Ground → Arduino
```

---

# Real-Time Deployment Pipeline

```text
Capture Frame
      ↓
Module 1
      ↓
Extract Grasps
      ↓
Compute Objectness
      ↓
Build State Vector
      ↓
PPO Policy
      ↓
Decision

WAIT
TRACK
REPOSITION
EXECUTE
ABORT

      ↓
Safety Gate
      ↓
Arduino Serial
      ↓
Motor Trigger
```

---

# Safety Mechanisms

## Objectness Gating

Prevents actions when no object is visible.

---

## Execute Safety Gate

Execute only when:

```text
Confidence High

Uncertainty Low

Disturbance Low

Stability High
```

---

## Hardware Execution Control

Motor activates only during:

```text
ACTION = EXECUTE
```

Motor stops immediately when action changes.

---

# Current Deployment Status

## Completed

* Module 1 Training
* Module 2 State Estimation
* Module 3 PPO Policy
* Raspberry Pi Deployment
* Pi Camera Integration
* Arduino Integration
* Serial Communication
* L298N Control
* Motor Triggering
* Execute Safety Gating
* Real-Time HUD


---

# Author

Arnav Karnik

MIT Manipal → IIT Jodhpur Internship Project

DG²-RL: Disturbance-Guided Grasp Decision Reinforcement Learning

