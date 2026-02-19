# Derivation of Special Residual and Jacobian for Question 1

## Problem Statement and Uniqueness

**IMPORTANT NOTE:** This problem contains an **artificial singularity** that requires special treatment. The mathematical derivation presented here is **NOT included in the main project report** due to its highly specialized nature and unique formulation.

---

## The Differential Equation

The boundary value problem is defined as:

$$
y'' = -\frac{y'}{x} + e^{20y}
$$

or equivalently:

$$
y'' + \frac{y'}{x} = e^{20y}
$$

### Boundary Conditions

- **Left boundary** (at $x = 0$): $y'(0) = 0$
- **Right boundary** (at $x = 1$): $y(1) = 0$

---

## The Artificial Singularity Problem

### Why This Problem is Unique

The term $\frac{y'}{x}$ creates an **artificial singularity** at $x = 0$. Standard finite difference methods fail at this point because:

1. The derivative $y'$ is divided by $x$
2. At the left boundary where $x = 0$, this term becomes undefined ($\frac{y'(0)}{0}$)

This singularity is called "artificial" because:

- The actual solution is well-behaved at $x = 0$
- Using the boundary condition $y'(0) = 0$, we have an indeterminate form $\frac{0}{0}$

---

## Derivation of the Special Residual at $x = 0$

### Step 1: Apply L'Hôpital's Rule

At $x = 0$, the differential equation is:

$$
y''(0) + \frac{y'(0)}{0} = e^{20y(0)}
$$

Since $y'(0) = 0$ (from boundary condition), the term $\frac{y'(0)}{0}$ is an indeterminate form $\frac{0}{0}$.

Applying L'Hôpital's rule:

$$
\lim_{x \to 0} \frac{y'(x)}{x} = \lim_{x \to 0} \frac{d(y')}{dx} = y''(0)
$$

So at $x = 0$, the differential equation becomes:

$$
y''(0) + y''(0) = e^{20y(0)}
$$

$$
2y''(0) = e^{20y(0)}
$$

$$
\boxed{y''(0) = \frac{e^{20y(0)}}{2}}
$$

### Step 2: Finite Difference Approximation

we apply central difference approximation to the neumann boundary condition present at $x =0$ :

$$
y'(0) \approx \frac{w_{1} - w_{-1}}{2h} = 0 \implies w_{1} = w_{-1}
$$

For the second derivative at $x = 0$, we again use central difference approximation.

$$
y''(0) \approx \frac{w_{1} - 2w_{0} + w_{-1}}{h^2}
$$

substituting $w_{1} = w_{-1}$

$$
y''(0) \approx \frac{2(w_{1} - w_{0})}{h^2}
$$

### Step 3: Formulation of the Residual

From the equation $y''(0) = \frac{e^{20y(0)}}{2}$:

$$
\frac{2(w_1 - w_0)}{h^2} = \frac{e^{20y(0)}}{2}
$$

Multiplying both sides by $h^2$:

$$
2(w_1 - w_0) = \frac{h^2}{2}e^{20y(0)}
$$

Rearranging:

$$
2w_1 - 2w_0 = \frac{h^2}{2}e^{20y(0)}
$$

The residual at $x = 0$ is:

$$
\boxed{R_0(w_0, w_1, h) = 2w_0 - 2w_1 + \frac{h^2}{2}e^{20y(0)}}
$$

Or equivalently:

$$
\boxed{R_0(w_0, w_1, h) = -2w_1 + 2w_0 + \frac{h^2}{2}e^{20y(0)}}
$$

## Derivation of the Special Jacobian at $x = 0$

The Jacobian matrix requires partial derivatives of the residual with respect to the unknowns.

### Residual Recap

$$
R_0 = 2w_0 - 2w_1 + \frac{h^2}{2}e^{20y(0)}
$$

### Jacobian Terms

#### Diagonal Term: $\frac{\partial R_0}{\partial w_0}$

$$
\frac{\partial R_0}{\partial w_0} = \frac{\partial}{\partial w_0}\left[2w_0 - 2w_1 + \frac{h^2}{2}e^{20y(0)}\right]
$$

$$
= 2 + \frac{h^2}{2} \cdot 20 \cdot e^{20y(0)}
$$

$$
\boxed{\frac{\partial R_0}{\partial w_0} = 2 + 10h^2e^{20y(0)}}
$$

#### Upper Diagonal Term: $\frac{\partial R_0}{\partial w_1}$

$$
\frac{\partial R_0}{\partial w_1} = \frac{\partial}{\partial w_1}\left[2w_0 - 2w_1 + \frac{h^2}{2}e^{20y(0)}\right]
$$

$$
\boxed{\frac{\partial R_0}{\partial w_1} = -2}
$$

### Jacobian Vector for Left Boundary

The special Jacobian at the left boundary is a tuple:

$$
\boxed{J_{\text{left}} = \left(2 + 10h^2e^{20y(0)}, \quad -2\right)}
$$

where:

- First element: diagonal contribution
- Second element: upper diagonal contribution

---

## Implementation in Code

### Special Residual Function

```python
bc_left_res = lambda w, h: -2*w[1] + 2*w[0] + np.exp(20*w[0])*h**2/2
```

### Special Jacobian Function

```python
bc_left_jac = lambda w, h: (2 + 10*h**2*np.exp(20*w[0]), -2)
```

---

## Why This Derivation is Not in the Main Report

1. **Highly Specialized**: This treatment is specific to problems with artificial singularities at boundary points
2. **Non-Standard Formulation**: The approach combines L'Hôpital's rule with finite differences in a unique way
3. **Implementation Detail**: The main report focuses on the general FDM methodology
4. **Uniqueness**: Most standard BVPs don't require this special handling
