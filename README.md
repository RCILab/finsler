# Finsler Flow Matching

**There and Back Are Not the Same: Finsler Flow Matching for Asymmetric Robot Planning**

Project page: **https://rcilab.khu.ac.kr/finsler/**
(`rcilab.github.io/finsler` redirects here)

A legged robot walks forward faster than it walks backward, and sideways slower still. Reversible
geometry cannot express that: a Riemannian metric measures `‖v‖ = ‖−v‖`. We take the gauge function of
the robot's measured velocity envelope, which is a non-reversible **Finsler metric** whose forward
distance is exactly the minimum travel time, use its geodesics as the conditional paths of a
flow-matching model, and thereby amortize time-optimal planning into a single ODE solve.

## What is here

| | |
|---|---|
| `index.html`, `assets/` | the project page: five videos, paper PDF, supplement PDF, code ZIP |
| `code/` | Finsler library, experiments, MuJoCo pipeline, numerical tests |

Videos on the page are generated from the code and reproduce the numbers reported in the paper exactly.

## Results at a glance

| | Euclidean | Symmetric Riemannian | **Finsler (ours)** |
|---|---:|---:|---:|
| SE(2) pose planning, vs exact geodesics | 1.212 | 1.064 | **0.998** |
| Planning from an envelope learned from noisy rollouts | — | 1.126 | **1.023** |
| Simulated H1-2, goal behind with heading reversed | — | 6.70 s | **4.76 s** |

Per-query planning latency is 13 ms in the plane and 29 ms on SE(2) on one CPU, against 330 ms per goal
plus a 9.9 s graph build for grid search and 75 s per query for a shooting solver. Because the field is
conditioned on the envelope, it also answers queries for capability it has never seen at the same cost,
where a grid solver must rebuild its graph for every new metric.

## Running the code

```bash
cd code
python tests/run_tests.py            # 34 numerical checks
python tests/run_tests_general.py    # 22
python tests/run_tests_se2.py        # 17
```

Requires `numpy`, `scipy`, `torch`, `matplotlib`; the simulation additionally needs `mujoco` and `pyyaml`
plus a clone of [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) under `code/sim/third_party/`.
`code/README.md` maps every table and figure of the paper to the script that produced it.

## Status

Simulation study; the quadruped hardware experiments are in progress. The paper is under review, so the
author list on the page is anonymous.

---

Robot Control and Intelligence Laboratory, Kyung Hee University · https://rcilab.khu.ac.kr
