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

- Core solver code: [Project2/Core](Project2/Core)
- Main entry point: [Project2/main.py](Project2/main.py)
- Configuration: [Project2/configs](Project2/configs)
- Outputs: [Project2/outputs](Project2/outputs)
- Analysis scripts: [Project2/analysis](Project2/analysis)
- Test scripts: [Project2/Test](Project2/Test)

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
