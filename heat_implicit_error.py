import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad
import time as time_module

a = 0.02  
k = 50  
H = 2  # h/k
h = 100  
alpha = 1.282051e-5 
Ti = 200  
To = 50 
Nx = 100  
dx = a / Nx 
dt = 0.001 

time_intervals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Numerical solution
def numerical_solution():
    print("Starting numerical (implicit) solution...")
    start_time = time_module.time()

    num_steps = int(max(time_intervals) / dt)
    T = np.ones(Nx + 1) * Ti
    x_numerical = np.linspace(0, a, Nx)

    A = np.zeros((Nx + 1, Nx + 1)) # Create coefficient matrix for implicit scheme
    b = np.zeros(Nx + 1)

    flux_profiles = {}
    time_steps = [int(t / dt) for t in time_intervals]

    for i in range(1, Nx):  # Coefficient matrix for implicit scheme
        A[i, i - 1] = -alpha * dt / dx ** 2  # -fo
        A[i, i] = 1 + 2 * alpha * dt / dx ** 2  # 1 + 2*fo
        A[i, i + 1] = -alpha * dt / dx ** 2  # -fo

    A[0, 0] = 1  
    A[0, 1] = -1

    A[Nx, Nx - 1] = -1 
    A[Nx, Nx] = 1 + (H * dx / alpha)

    for t in range(1, num_steps + 1):
        for i in range(1, Nx):
            b[i] = T[i]

        b[0] = 0  # BC at x=0
        b[Nx] = (H * dx / alpha) * To  # BC at x=a

        T_new = solve_heat_equation(A, b)
        T = T_new

        if t in time_steps: # Calculate and store flux at specified time intervals
            time_idx = time_steps.index(t)
            actual_time = time_intervals[time_idx]

            q_dot = np.zeros(Nx)
            for i in range(1, Nx - 1):
                q_dot[i] = -k * (T[i + 1] - T[i]) / dx

            q_dot[0] = -k * (T[1] - T[0]) / dx  # Forward difference at left boundary
            q_dot[Nx - 1] = -k * (T[Nx] - T[Nx - 1]) / dx  # Forward difference at right boundary

            flux_profiles[int(actual_time)] = q_dot.copy()

    end_time = time_module.time()
    print(f"Numerical solution completed in {end_time - start_time:.4f} seconds")

    return x_numerical, flux_profiles

# TDMA solver
def solve_heat_equation(A, b):
    n = len(b)
    A_copy = A.copy()
    b_copy = b.copy()

    for k in range(n - 1):  # LU decomposition
        if abs(A_copy[k, k]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")

        for i in range(k + 1, n):
            factor = A_copy[i, k] / A_copy[k, k]
            A_copy[i, k + 1:] = A_copy[i, k + 1:] - factor * A_copy[k, k + 1:]
            A_copy[i, k] = factor

    y = np.zeros(n) # Forward substitution (Ly = b)
    y[0] = b_copy[0]
    for i in range(1, n):
        y[i] = b_copy[i]
        for j in range(i):
            y[i] -= A_copy[i, j] * y[j]

    x = np.zeros(n) # Back substitution (Ux = y)
    x[n - 1] = y[n - 1] / A_copy[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= A_copy[i, j] * x[j]
        x[i] /= A_copy[i, i]

    return x

def analytical_solution(x_points):
    print("Starting analytical solution...")
    start_time = time_module.time()
    
    N = 50
    
    def lambda_eq(lambda_n):
        return np.tan(lambda_n * a) - (lambda_n * k / h)

    lambda_n = np.array([fsolve(lambda_eq, (n * np.pi) / (2 * a))[0] for n in range(1, N + 1)])

    flux_profiles = {}

    for t in time_intervals:
        q_dot = np.zeros_like(x_points)
        for x_idx, x_val in enumerate(x_points):
            q_val = k * To 
            for n in range(N):
                coeff, _ = quad(lambda x: (Ti - To) * np.sin(lambda_n[n] * x), 0, a)
                coeff *= 2 / a
                term = -k * coeff * lambda_n[n] * np.exp(-(lambda_n[n] ** 2) * alpha * t) * np.sin(lambda_n[n] * x_val)
                q_val -= term
            q_dot[x_idx] = q_val
            
        flux_profiles[int(t)] = q_dot

    end_time = time_module.time()
    print(f"Analytical solution completed in {end_time - start_time:.4f} seconds")

    return flux_profiles

def main():
    x_numerical, numerical_flux = numerical_solution()
    analytical_flux = analytical_solution(x_numerical)
    print("Numerical flux keys:", sorted(list(numerical_flux.keys())))
    print("Analytical flux keys:", sorted(list(analytical_flux.keys())))
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_intervals)))
    
    for i, t in enumerate(time_intervals):
        t_int = int(t)
        if t_int not in numerical_flux or t_int not in analytical_flux:
            print(f"Warning: Time {t_int} not found in one of the dictionaries.")
            continue
        error = np.abs(analytical_flux[t_int] - numerical_flux[t_int])
        plt.plot(x_numerical, error, color=colors[i], label=f't = {t_int} s')

    plt.xlabel('Position (m)')
    plt.ylabel('Absolute Error in Heat Flux (W/m²)')
    plt.title('Absolute Error Between Analytical and Implicit Numerical Heat Flux Solutions')
    plt.legend()
    plt.grid(True)
    plt.show()


#if __name__ == "__main__":
    #main()
