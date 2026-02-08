from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

@njit
def f_2(x, u_0, u_1):
    return -(u_1**2)/(u_0+1e-4)
    # return 4*u_0**2
    # return -u_0

@njit
def f_1(x, u_0, u_1):
    return u_1

@njit
def RK4(x, nx, h, y_0, ydash_0):
    u_0 = np.zeros(nx+1)
    u_1 = np.zeros(nx+1)
    u_0[0] = y_0
    u_1[0] = ydash_0
    for i in range(nx):
        u0_i = u_0[i]
        u1_i = u_1[i]
        # k1
        k1_1 = h * f_1(x[i], u0_i, u1_i)
        k1_2 = h * f_2(x[i], u0_i, u1_i)
        # k2 (half step)
        u0_k2 = u0_i + 0.5 * k1_1
        u1_k2 = u1_i + 0.5 * k1_2
        k2_1 = h * f_1(x[i] + 0.5*h, u0_k2, u1_k2)
        k2_2 = h * f_2(x[i] + 0.5*h, u0_k2, u1_k2)
        # k3 (half step)
        u0_k3 = u0_i + 0.5 * k2_1
        u1_k3 = u1_i + 0.5 * k2_2
        k3_1 = h * f_1(x[i] + 0.5*h, u0_k3, u1_k3)
        k3_2 = h * f_2(x[i] + 0.5*h, u0_k3, u1_k3)
        # k4 (full step)
        u0_k4 = u0_i + k3_1
        u1_k4 = u1_i + k3_2
        k4_1 = h * f_1(x[i] + h, u0_k4, u1_k4)
        k4_2 = h * f_2(x[i] + h, u0_k4, u1_k4)
        u_0[i+1] = u0_i + (k1_1 + 2*k2_1 + 2*k3_1 + k4_1) / 6.0
        u_1[i+1] = u1_i + (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6.0
    return u_0, u_1

def SecantMethod(x, nx, h, y_0, ydash1_0, ydash2_0, y_1, tol=1e-6, N=100):
    y1_u0, y1_u1 = RK4(x, nx, h, y_0, ydash1_0)
    if (abs(y1_u0[nx] - y_1) < tol):
        return y1_u0, y1_u1
    y2_u0, y2_u1 = RK4(x, nx, h, y_0, ydash2_0)
    if (y2_u0[nx] == y_1):
        return y2_u0, y2_u1
    no_of_iterations = 0
    for _ in tqdm(range(N), desc="Solving ODE"):
        if (abs(y_1 - y2_u0[nx]) <= tol):
            break
        ydash_new = ydash2_0 - ((y2_u0[nx] - y_1) * (ydash2_0 - ydash1_0)) / (y2_u0[nx] - y1_u0[nx])
        y_final_u0, y_final_u1 = RK4(x, nx, h, y_0, ydash_new)
        y1_u0, y1_u1 = y2_u0, y2_u1
        y2_u0, y2_u1 = y_final_u0, y_final_u1
        ydash1_0 = ydash2_0
        ydash2_0 = ydash_new
        no_of_iterations += 1
    print("No of iterations: ", no_of_iterations)
    return y2_u0, y2_u1

# Driver code (copied from notebook)
if __name__ == "__main__":
    y_exact = lambda x: 0.0001*((100020000*x+1)**0.5-1)
    y_0 = 0
    y_1 = 1
    ydash1_0 = 0.5
    ydash2_0 = 1.5
    h = 1e-8
    start = 0
    end = 1
    nx = (int)((end-start)/h)
    x = np.linspace(start, end, nx+1, dtype=np.float32)
    y_analytical = y_exact(x)
    y_reference, _ = SecantMethod(x, nx, h, y_0, ydash1_0, ydash2_0, y_1, 1e-6, 100)
    
    # Save y_reference[::100] to file
    output_path = os.path.join(os.path.dirname(__file__), "y_ref.txt")
    np.savetxt(output_path, y_reference[::100])
    
    print(x)
    print(y_reference[::1000])

    plt.plot(x[::1000], y_analytical[::1000], label = "Analytical Solution")
    plt.grid()
    plt.plot(x[::1000], y_reference[::1000], label = "Reference Solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Exact vs. Reference Solution")
    plot_path1 = os.path.join(os.path.dirname(__file__), "Analytical vs. Reference.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(y_analytical[::1000])
    plt.figure(figsize=(10, 6))
    plt.plot(x[::1000], abs(y_analytical[::1000]-y_reference[::1000]), label = "Error")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.title("Error between Analytical and Reference Solution")
    plot_path2 = os.path.join(os.path.dirname(__file__), "Absolute Error.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.show()
    max_error = max(abs(y_analytical[::1000]-y_reference[::1000]))
    print(max_error)
