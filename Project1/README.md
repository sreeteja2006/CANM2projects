# BVP Solver using Shooting Method

Solves boundary value problems for second-order ODEs. Written for CANM-2.

## What's this?

Given an ODE like $y'' + 2y' + 3y = 0$ with conditions at both ends (say $y(0)=1$ and $y(1)=2$), this solver uses the shooting method to find the solution numerically.

## Project Structure

```
Project1/
├── input.txt           # Your problem goes here
├── README.md
└── ode_solver/
    ├── main.py         # Run this
    ├── core/           # Shooting method implementation
    ├── methods/        # RK2, RK4, ABM2, ABM4
    ├── analysis/       # Convergence & stability plots
    └── Plots/          # Output plots saved here
```

## Quick Start

1. Install dependencies:

   ```bash
   pip install numpy matplotlib tqdm
   ```

2. Run from the ode_solver folder:

   ```bash
   cd Project1/ode_solver
   python main.py
   ```

3. Pick option 1 to use `input.txt`, or option 2 to enter values manually.

## Setting up input.txt

The file has comments explaining each line. Here's the format:

```
2                           # Order of ODE (1 or 2)
[y', -2*y' - 3*y]          # The ODE as a system
0                           # x start
1                           # x end
0.01                        # step size
1, 0, -1                    # BC at start: ay + by' + c = 0
1, 0, -2                    # BC at end
0, 1                        # two guesses for shooting
1                           # method (1=RK4, 2=RK2, 3=ABM2, 4=ABM4)
n                           # stability plots?
n                           # convergence plots?
```

## Writing the ODE

Convert your second-order ODE to first-order form:

$$y'' = f(x, y, y') \rightarrow [y', f(x,y,y')]$$

Examples:

- $y'' + 2y' + 3y = 0$ becomes `[y', -2*y' - 3*y]`
- $y'' = -y$ becomes `[y', -y]`
- $y'' = x + y'^2$ becomes `[y', x + y'**2]`

You can use `y` and `y'` directly, or `u[0]` and `u[1]` if you prefer.

## Boundary Conditions

Format: $ay + by' + c = 0$

| What you want | a | b | c |
|---------------|---|---|---|
| $y(0) = 1$ | 1 | 0 | -1 |
| $y'(0) = 2$ | 0 | 1 | -2 |
| $y(1) = 0$ | 1 | 0 | 0 |

## Methods

1. **RK4** - 4th order Runge-Kutta (default, most accurate)
2. **RK2** - 2nd order Runge-Kutta
3. **ABM2** - Adams-Bashforth-Moulton 2nd order
4. **ABM4** - Adams-Bashforth-Moulton 4th order

## If things go wrong

- **Blows up**: Use smaller step size or try different guesses
- **Won't converge**: Your guesses might be too close together, or the problem is stiff
- **Syntax error**: Check your function - use `**` for powers, match brackets

## How shooting method works

We don't know $y'(0)$, so we guess. Solve the IVP with that guess, check if we hit the right value at $x=1$. If not, adjust the guess (using secant method) and try again. Repeat until it works.

---

CANM-2 Project-1
