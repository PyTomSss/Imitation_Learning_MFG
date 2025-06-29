# Imitation Learning in Mean-Field Games (MFG-IL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This repository contains Python implementations and illustrative notebooks for exploring **Imitation Learning in Mean-Field Games (MFGs)**. The work focuses on theoretical guarantees and empirical comparisons of different imitation learning approaches, including **Behavioral Cloning**, **Adversarial Imitation (VANILLA)**, and **Model Predictive Control (MPC-based)** learning proxies.

> 📖 For full theoretical background and derivations, see the companion paper [`Interactions_MVA.pdf`](./Interactions_MVA.pdf).

---

## 🧠 Context

In Mean-Field Games, agents interact indirectly through a population distribution. Learning in such settings is fundamentally more difficult than in classical single-agent environments. This project introduces a **new metric**, the **Nash Imitation Gap (NIG)**, to measure how far a policy is from being an equilibrium in population.

---

## 📁 Repository Structure

```text
.
├── Crowd_Avoidance_MFG.py     # 2-state congestion game
├── MFG_attractor.py           # absorbing-state attractor MFG
├── Tri_Congestion_MFG.py      # 3-state congestion-dependent MFG
├── utils.py                   # plotting utilities (Pareto curves, error evolution)
├── toy_examples.ipynb         # quick start notebook with simulations
├── Interactions_MVA.pdf       # full paper with theoretical results
├── LICENSE                    # MIT License
└── README.md                  # this file

```

## 🧪 Toy Environments

Three toy MFGs are implemented and used to benchmark imitation errors and policy robustness:

- `CrowdAvoidanceMFG` – models congestion with decreasing transition success.

- `AttractorMFG` – illustrates accumulation in a cost-heavy absorbing state.

- `TriStateCongestionMFG` – a three-state MFG with indirect imitation and congestion.

Each environment provides methods to:

- simulate state distributions under expert and learner policies (run)

- compute proxy errors (errors)

- compute the Nash Imitation Gap (NIG)

## 📊 Experiments

You can visualize the performance of different imitation proxies via:

### 1. *Pareto Frontier*

```
python
Copier
Modifier
from utils import plot_pareto_eps_vs_nig

plot_pareto_eps_vs_nig(
    model_cls=CrowdAvoidanceMFG,
    param_grid=np.linspace(0, 1, 100),
    horizon=40,
    extra_kwargs={"L": 0.8}
)
```

### 2. *Time Evolution of Errors*

```
python
from utils import plot_error_curves

plot_error_curves(
    model_cls=TriStateCongestionMFG,
    alphas=(0.4, 0.5),
    horizon=30,
    extra_kwargs={"L1": 0.5, "L2": 0.7}
)
```
-💡 `See toy_examples.ipynb` for ready-to-run demos.

## 📖 Theory & Metrics

We define and study three imitation error proxies:

- *Behavioral Cloning* (BC): state-wise action matching.

- *Vanilla Adversarial* (VAN): occupancy matching using expert dynamics.

- *MPC-based ADV* (MFC): occupancy matching using learner-induced dynamics.

Each method has a closed-form expression for error computation and is benchmarked across a range of policies and horizon lengths.

## 📚 Reference

The present project was strongly inspired from the work **On Imitation in Mean-field Games** (Ramponi et al.).

## Authors 

- Tom Rossa
- Naïl Khelifa
