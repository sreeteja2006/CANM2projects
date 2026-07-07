# CANM2projects
Two numerical methods coursework projects for solving boundary value problems.

## What it does
Project1 solves second-order ODE boundary value problems with the shooting method. It supports RK2, RK4, ABM2, and ABM4 for the underlying IVP solve.

Project2 solves nonlinear second-order boundary value problems with a finite difference method and Newton iteration. It can read problem settings from a config file or from interactive input, and it can save plots and CSV output.

## Installation
From the repository root:

```bash
cd /home/tanish/CANM2projects
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use the smaller dependency set for Project2, install that file instead:

```bash
pip install -r Project2/requirements.txt
```

## Usage
Run Project1 from its solver directory:

```bash
cd /home/tanish/CANM2projects/Project1/ode_solver
python main.py
```

The program reads `input.txt` when you choose file input, or it can prompt for the ODE, domain, boundary conditions, guesses, and method.

Run Project2 from the project folder:

```bash
cd /home/tanish/CANM2projects/Project2
python main.py
```

The program can load `configs/config.json` or accept the problem data interactively.

## Requirements/Dependencies
Python 3 and the packages listed in `requirements.txt` and `Project2/requirements.txt`.

The main runtime libraries are `numpy`, `matplotlib`, and `scipy`. Project2 also uses `numba` and `pandas`.

## Configuration
Project1 uses `Project1/input.txt` for file-based problem setup.

Project2 uses `Project2/configs/config.json` for its default configuration.

