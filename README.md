# SteadyState
A small project that analyzes the **convergence of several Markov Chains**.

## Overview
- We first examine **two Markov Chains** with significantly different transition
matrices in [`convergence.py`](steadystate/convergence.py).
- Then, we extend the analysis to **three additional chains** with distinct
behaviors in [`extra_transitions.py`](steadystate/extra_transitions.py).
- We compare how **different initial distributions** affect convergence in
[`extra_initial_dists.py`](steadystate/extra_initial_dists.py).
- Finally, we **simulate** Markov Chains step by step to observe empirical
behavior in [`simulation.py`](steadystate/simulation.py).

This project is part of my **Bayesian Networks and Hidden Markov Models** class.

Built together with my colleague **Mara Fodor**.

---

## **Setup**
### **Poetry (Recommended)**
The project uses [Poetry](https://python-poetry.org/) for dependency management.

After installing Poetry:

1. **Install project dependencies:**
   ```sh
   poetry install
   ```

2. **Activate the project environment:**
   ```sh
   poetry shell
   ```

3. **Run the different scripts:**
   ```sh
   python -m steadystate.convergence
   python -m steadystate.extra_transitions
   python -m steadystate.extra_initial_dists
   python -m steadystate.simulation
   ```
