 # 📌 **Vision-Guided Adaptive Grasp Decision Making for Aerial Manipulation**

### DG²-RL: Disturbance-Guided Grasp Decision Reinforcement Learning

A real-time hierarchical reinforcement learning framework for disturbance-aware robotic grasp decision making under dynamic conditions.

---

# Overview

DG²-RL is a disturbance-aware robotic grasping framework designed for real-time autonomous grasp execution under unstable environmental conditions such as:

* aerial manipulation
* drone-based grasping
* moving targets
* vibration disturbances
* unstable camera motion
* uncertain grasp predictions

The framework combines:

1. Multi-grasp prediction network
2. Disturbance estimation and temporal stability reasoning
3. PPO-based hierarchical reinforcement learning policy
4. Hardware-triggered grasp execution

The system performs real-time perception, disturbance reasoning, action selection, and physical actuation using a servo-driven grasp execution pipeline.

---

# Pipeline Architecture

```text
RGB Camera
     ↓
Module 1: Multi-Grasp Prediction
     ↓
Module 2: Disturbance & Temporal Estimation
     ↓
DG²-RL PPO Meta-Decision Policy
     ↓
Action Selection
(WAIT / TRACK / REPOSITION / EXECUTE / ABORT)
     ↓
Hardware Execution
(Arduino + PCA9685 + Servo)
```

---

# Key Features

* Real-time grasp prediction
* Disturbance-aware reinforcement learning
* Temporal stability modeling
* Optical-flow-based motion estimation
* Hardware-in-the-loop execution
* PPO-based hierarchical control
* Safety-aware execution logic
* Object-aware policy gating
* Servo-triggered physical execution
* Live HUD telemetry visualization

---

# Action Space

The DG²-RL policy operates over five high-level actions:

| Action     | Description             |
| ---------- | ----------------------- |
| WAIT       | Wait for stabilization  |
| TRACK      | Track moving target     |
| REPOSITION | Adjust alignment        |
| EXECUTE    | Trigger grasp execution |
| ABORT      | Abort unsafe grasp      |

---

# State Representation

The PPO policy receives a 10D disturbance-aware state vector:

| Feature   | Description              |
| --------- | ------------------------ |
| q_max     | Maximum grasp confidence |
| sigma_max | Grasp uncertainty        |
| A_t       | Disturbance amplitude    |
| f_t       | Disturbance frequency    |
| e_p       | Positional error         |
| ep_dot    | Positional error rate    |
| S_t       | Temporal stability       |
| delta_q   | Confidence variation     |
| v_d       | Drone/camera velocity    |
| omega     | Motion direction         |

---

# Hardware Stack

## Controller

* Arduino UNO R4 Minima

## Servo Driver

* PCA9685 PWM Servo Driver

## Actuator

* Servo motor (connected to PCA9685 Channel 15)

## Sensor

* USB webcam

---

# Software Stack

* Python
* PyTorch
* OpenCV
* NumPy
* Arduino
* PPO Reinforcement Learning
* PCA9685 Servo Control

---

# Real-Time Features

The system supports:

* live webcam inference
* dynamic action selection
* real-time HUD overlays
* servo-triggered execution
* reward monitoring
* temporal stability tracking
* policy debugging

---

# Experimental Results

The framework demonstrates:

* stable temporal reasoning
* disturbance-aware decision making
* successful hardware-triggered execution
* conservative safety-oriented policy behavior
* successful PPO integration

Example metrics from live testing:

| Metric               | Value  |
| -------------------- | ------ |
| Avg Reward           | 1.343  |
| Max Reward           | 16.851 |
| Avg Confidence       | 0.320  |
| Avg Stability        | 0.775  |
| Execute Success Rate | 100%   |

---

# Observed Policy Behavior

The learned policy exhibits:

* WAIT dominance during unstable conditions
* TRACK during motion disturbances
* EXECUTE during stable high-confidence states
* ABORT during unsafe conditions

This demonstrates meaningful disturbance-aware policy learning rather than random action selection.

---

# Hardware Execution Proof

The complete robotic execution pipeline was validated:

```text
DG²-RL Policy
    ↓
Serial Communication
    ↓
Arduino Controller
    ↓
PCA9685 Driver
    ↓
Servo Actuation
```

Real-time EXECUTE actions successfully triggered physical servo motion.

---

# Current Limitations

* Webcam-only depth approximation
* Limited object-awareness under empty scenes
* No real aerial deployment yet
* Disturbance frequency normalization requires further calibration

---

# Research Contributions

DG²-RL introduces:

* disturbance-guided hierarchical RL for grasping
* temporal stability-aware execution intelligence
* uncertainty-aware meta-decision control
* real-time hardware-integrated RL grasp execution


---

# License

MIT License

---

# Acknowledgements

Developed as part of research work involving:

* IIT Jodhpur
* Manipal Institute of Technology

---
