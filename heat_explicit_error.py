import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad

a = 0.02 
k = 50  
H = 2  # h/k
h = 100 
alpha = 1.282051e-5 
Ti = 200  
To = 50 
time_intervals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] 


# Numerical (explicit) solution
def numerical_solution():
    Nx = 100 
    dx = a / Nx 
    dt = 0.001

    num_steps = int(max(time_intervals) / dt
    T = np.ones(Nx) * Ti
    
    flux_profiles = {}
    time_steps = [int(t / dt) for t in time_intervals]
    x_numerical = np.linspace(0, a, Nx)

    for n in range(1, num_steps + 1):
        T_new = T.copy()

        for i in range(1, Nx - 1):
            T_new[i] = T[i] + Fo * (T[i + 1] - 2 * T[i] + T[i - 1])

        T_new[0] = T_new[1] 
        T_new[Nx - 1] = (T[Nx - 2] + (H * dx / alpha) * To) / (1 + H * dx / alpha)

        T = T_new

        if n in time_steps:
            time_idx = time_steps.index(n)
            actual_time = time_intervals[time_idx]  
            q_dot = np.zeros(Nx)
            for i in range(1, Nx - 1):
                q_dot[i] = -k * (T[i + 1] - T[i - 1]) / (2 * dx)
            q_dot[0] = -k * (T[1] - T[0]) / dx  # Forward difference at the left boundary
            q_dot[Nx - 1] = -k * (T[Nx - 1] - T[Nx - 2]) / dx  # Backward difference at the right boundary
            
            flux_profiles[int(actual_time)] = q_dot.copy()

    return x_numerical, flux_profiles


# Analytical solution
def analytical_solution(x_points):
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
            
        flux_profiles[int(t)] = q_dot.copy()

    return flux_profiles


# Calculate the error
def main():
    x_numerical, numerical_flux = numerical_solution()
    analytical_flux = analytical_solution(x_numerical)

    print("Numerical flux keys:", list(numerical_flux.keys()))
    print("Analytical flux keys:", list(analytical_flux.keys()))

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_intervals)))

    for i, t in enumerate(time_intervals):
        t_int = int(t)   
        error = np.abs(analytical_flux[t_int] - numerical_flux[t_int])
        plt.plot(x_numerical, error, color=colors[i], label=f't = {t_int} s')

    plt.xlabel('Position (m)')
    plt.ylabel('Absolute Error in Heat Flux (W/m²)')
    plt.title('Absolute Error Between Analytical and Numerical (explicit) Heat Flux Solutions')
    plt.legend()
    plt.grid(True)
    plt.show()


#if __name__ == "__main__":
    #main()
