# Portfolio Exam 2 — Particle Filter / Sensor Fusion

**Course:** Reasoning and Decision Making under Uncertainty
**University:** TH Würzburg-Schweinfurt
**Due:** 26.06.2026

---

## What this does

Estimates the **positions and velocity vectors** of `n ≥ 1` simultaneously flying balls
using a Particle Filter. Only noisy, intermittent sensor readings are available —
the true launch parameters (position, speed, angle) are completely unknown to the filter.

---

## Setup

> **Prerequisite:** [Poetry](https://python-poetry.org) must be installed.

### Step 1 — Install project dependencies

Inside the `portfolio2/` directory:

```bash
poetry install
```

This creates a project-local virtual environment and installs all dependencies
(numpy, matplotlib, scikit-learn, PyQt5).

### Step 2 — Run

```bash
poetry run python3 main.py
```

A window opens with the live animation. Close it to exit.

> **Optional — activate the environment in your shell:**
> ```bash
> eval $(poetry env activate)
> ```
> After this you can call `python3 main.py` directly without the `poetry run` prefix.

---

## Project structure

```
portfolio2/
├── config.py           # All tunable parameters in one place
├── models.py           # Ball and Particle data classes
├── simulation.py       # Ground-truth physics + noisy sensor
├── particle_filter.py  # Particle Filter: predict / update / resample / cluster
├── visualization.py    # Real-time matplotlib animation
├── main.py             # Entry point
├── pyproject.toml      # Poetry project file (dependencies, Python version)
├── pyrightconfig.json  # IDE type-checker config (for VS Code / Pylance)
└── README.md
```

---

## Key parameters (`config.py`)

Change these to experiment without touching any logic:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `NUM_BALLS` | `3` | Number of balls flying at the same time |
| `NUM_PARTICLES` | `4000` | More → better accuracy, slower animation |
| `POSITION_NOISE_STD` | `1.5 m` | How noisy the sensor measurements are |
| `OBSERVATION_INTERVAL` | `3` | Sensor fires every N simulation steps |
| `DROP_START / DROP_END` | `50 / 90` | Step range where sensor goes completely silent |
| `TOTAL_TIME` | `8.0 s` | Total simulation duration |
| `DBSCAN_EPS` | `2.5 m` | Neighborhood radius for ball cluster detection |
| `SEED` | `42` | Random seed — change to get different trajectories |

---

## How the Particle Filter works

### State per particle

Each particle is one hypothesis about a single ball:

```
[ x,  y,  vx,  vy ]
  pos      velocity
```

### Algorithm — one cycle per timestep

**1. Predict** (transition model)

Move every particle forward using projectile physics plus small noise:

```
x  ← x  + vx · dt  + noise
y  ← y  + vy · dt  + noise
vx ← vx             + noise
vy ← vy − g · dt   + noise
```

Gravity (`−g · dt`) acts on the vertical velocity each tick.
The noise lets particles adapt to small deviations from ideal ballistics.

**2. Update** (observation / likelihood model)

Score each particle against the sensor reading using a Gaussian:

```
likelihood = exp( −d² / 2σ² )
```

where `d` is the distance to the **closest** observation.
Using "closest" (not assigned) is essential — the balls are indistinguishable,
so the filter cannot know which measurement belongs to which ball.

**3. Normalize**

All weights are scaled to sum to 1.

**4. Resample** (when needed)

When the Effective Sample Size (ESS) drops below 50 % of particle count,
systematic resampling is triggered: high-weight particles are duplicated,
low-weight particles are discarded. This prevents degeneracy.

### Multiple balls → multimodal density → DBSCAN

Because each particle tracks one ball independently, the cloud naturally
forms one cluster per flying ball. After each step **DBSCAN** finds those
clusters without needing to know `n` in advance. Each cluster's
weighted mean becomes one estimate:

```
{ x, y, vx, vy }
```

This works whether the balls are flying close together or far apart.

### Sensor dropout

During the dropout window (steps `DROP_START` … `DROP_END`) no observations
arrive. The filter runs only the predict step, so particles follow ballistic
trajectories and the estimates remain available throughout the outage.

---

## Visualization legend

| Symbol | Meaning |
|--------|---------|
| Gray dots | Particle cloud (full posterior distribution) |
| Green circle | True ball position (ground truth, hidden from filter) |
| Blue + | Noisy sensor observation |
| Red X | Filter position estimate (one per detected ball) |
| Red arrow | Estimated velocity vector |
| Black line | Ground level (y = 0) |

The status bar at the top shows:
- current step number
- sensor status (`OK` or `DROPOUT`)
- current ESS value
- number of balls detected by DBSCAN

---

## Requirements

- Python 3.12+
- Poetry (for dependency management)
- A display server (X11 / Wayland) for the animation window
