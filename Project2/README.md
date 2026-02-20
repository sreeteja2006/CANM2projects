# Project2: Nonlinear BVP Solver (FDM)

This project solves nonlinear boundary value problems of the form $y'' = F(x, y, y')$ using a finite difference method with Newton's iteration. It only supports 2nd order ODEs.

## Quick Start

From the repository root:

```bash
cd Project2
python main.py
```

The CLI will guide you to either load a configuration file or enter parameters interactively.

## Configuration

- Default config path: [Project2/configs/config.json](Project2/configs/config.json)
- Update problem definition, domain, boundary conditions, functions, and solver settings in that file.
- If you move the config, update the path used in [Project2/main.py](Project2/main.py).

## Output Locations

- Plots: [Project2/outputs/plots](Project2/outputs/plots)
- CSV data: [Project2/outputs/data](Project2/outputs/data)

The main program saves numerical/analytical plots and optional CSV exports to these folders.

## Folder Structure (Key Parts)

- Core solver code: [Project2/core](Project2/core)
- Main entry point: [Project2/main.py](Project2/main.py)
- Configuration: [Project2/configs](Project2/configs)
- Outputs: [Project2/outputs](Project2/outputs)
- Analysis scripts: [Project2/analysis](Project2/analysis)
- Comparison scripts: [Project2/comparisons](Project2/comparisons)
- Test scripts: [Project2/Test](Project2/Test)

## Question1 Folder

[Project2/question1](Project2/question1) contains the special-case solver and derivation for Question 1. This problem includes an artificial singularity at $x = 0$, so the residual and Jacobian are derived separately and are not part of the general solver workflow.

- Solver implementation: [Project2/question1/q1_solver.py](Project2/question1/q1_solver.py)
- Derivation notes: [Project2/question1/Derivation.md](Project2/question1/Derivation.md)

### Usage (Brief)

From the Project2 folder:

```bash
python question1/q1_solver.py
```

To import the solver function:

```python
from question1.q1_solver import question_1_solver
```

## Comparisons Folder

[Project2/comparisons](Project2/comparisons) contains scripts to compare the accuracy, convergence, and stability characteristics of the Finite Difference Method (FDM) with the Shooting Method.

The comparison modules include:

- **Accuracy Comparison** ([Project2/comparisons/Accuracy_Comp.py](Project2/comparisons/Accuracy_Comp.py)): Compares the numerical accuracy of FDM against analytical and shooting method solutions. Uses numba-optimized RK2, RK4, and ABM schemes for efficient computation.

- **Convergence Comparison** ([Project2/comparisons/Convergence_compare.py](Project2/comparisons/Convergence_compare.py)): Analyzes convergence behavior by studying solution quality as grid density increases. Creates convergence plots to visualize both FDM and shooting method performance.

- **Stability Comparison** ([Project2/comparisons/stability_comp.py](Project2/comparisons/stability_comp.py)): Examines numerical stability of both methods by comparing condition numbers, singular values, and amplification factors across iterations.

### Usage

These comparison scripts can be run independently to analyze method behavior:

```python
from comparisons.stability_comp import StabilityComparison
from comparisons.Accuracy_Comp import *
from comparisons.Convergence_compare import *
```

Or integrate them into your workflow to understand how FDM and Shooting methods perform on your specific problem.

## Using FDM_Solver with Custom Jacobian and Residual

If you have your own Jacobian and Residual classes, pass them directly and omit `F`:

```python
from Core.FDM_Solver import FDM_Solver
from Core.Jacobian import Jacobian
from Core.Residual import Residual

# Build custom Jacobian/Residual objects
J = Jacobian(...)
R = Residual(...)

solver = FDM_Solver(N=N, domain=domain, BC=BC, jacobian=J, residual=R)
w = solver.solver(tol=1e-8, max_iter=100)
```

If you provide `F`, the solver will build its own Jacobian/Residual internally.

## Requirements

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The Project2 code imports `numpy` and `matplotlib` (plus Python standard library).

## Unused Libraries in requirements.txt

The Project2 source does not import most of the Jupyter/IPython stack listed in [requirements.txt](requirements.txt). It also does not use `tqdm`. These entries are safe to remove unless you plan to run notebooks.
