import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad

k = 50  
rho = 7800  
c_p = 500  
alpha = 1.282051e-5  
a = 0.02  
T_inf = 50 
Ti = 200 
h = 100
N = 50 
Nx = 100  
dx = a / Nx  
dt = 0.0001 
H = 2  # h/k
Fo = alpha * dt / dx**2

if Fo > 0.5:
    raise ValueError(f"Stability condition violated! Fo = {Fo}. Reduce dt or increase Nx.")

def lambda_eq(lambda_n):
    return np.tan(lambda_n * a) - (lambda_n * k / h)

lambda_n = np.array([fsolve(lambda_eq, (n * np.pi) / (2 * a))[0] for n in range(1, N + 1)])

x = np.linspace(0, a, Nx + 1)
time_intervals = np.arange(5, 55, 5).tolist()  # Convert to list for consistent indexing

analytical_solutions = {} # Compute analytical solution
for t in time_intervals:
    T = np.ones_like(x) * T_inf
    for n in range(N):
        integrand = lambda x: (Ti - T_inf) * np.sin(lambda_n[n] * x)
        coeff, _ = quad(integrand, 0, a)
        coeff *= (2 / a)
        T += coeff * np.exp(-(lambda_n[n]) ** 2 * alpha * t) * np.cos(lambda_n[n] * x)
    analytical_solutions[int(t)] = T.copy()

num_steps = int(max(time_intervals) / dt)  # Compute explicit numerical solution
T_profiles = {}
T = np.ones(Nx + 1) * Ti
for n in range(1, num_steps + 1, 1):
    T_new = T.copy()
    for i in range(1, Nx):
        T_new[i] = T[i] + Fo * (T[i + 1] - 2 * T[i] + T[i - 1])
    T_new[0] = T[1]
    T_new[Nx] = (T[Nx - 1] + (H * dx / alpha) * T_inf) / (1 + (H * dx / alpha))
    T = T_new
    if round(n * dt, 4) in time_intervals:  # Ensure floating point consistency
        T_profiles[int(round(n * dt, 4))] = T.copy()

plt.figure(figsize=(8, 6)) # Compute and plot error
for t in time_intervals:
    if int(t) in analytical_solutions and int(t) in T_profiles:
        error = np.abs(analytical_solutions[int(t)] - T_profiles[int(t)])
        plt.plot(x, error, label=f'Error at t={t}s')
    else:
        print(f"Warning: Missing data for t={t}s")

plt.xlabel('Position (m)')
plt.ylabel('Error (°C)')
plt.title('Error between Analytical and Numerical (explicit) Solution')
plt.legend()
plt.grid()
plt.show()
