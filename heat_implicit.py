import numpy as np
import matplotlib.pyplot as plt
import time as tme

alpha = 1.282051e-5
a = 0.02
Nx = 100  
dx = a / Nx
dt = 0.001
H = 2
k = 50
Ti = 200
To = 50
num_steps = int(50 // dt)

fo = alpha * dt / dx ** 2
x = np.linspace(0, a, Nx)  
T = np.ones(Nx + 1) * Ti  

start_time = tme.time() 

A = np.zeros((Nx + 1, Nx + 1)) # Coefficient matrix for implicit scheme (tdma)
b = np.zeros((Nx + 1))

T_profiles = {}
flux_profiles = {}

time_intervals = np.arange(5, 51, 5)
time_steps = [int(t / dt) for t in time_intervals]

for i in range(1, Nx): # constructing the matrix
    A[i, i - 1] = -fo
    A[i, i] = 1 + 2 * fo
    A[i, i + 1] = -fo

A[0, 0] = 1  
A[0, 1] = -1

A[Nx, Nx - 1] = -1  
A[Nx, Nx] = 1 + (H * dx / alpha)

def solve_heat_equation(A, b):
    n = len(b)
    A = A.copy()
    b = b.copy()

    for k in range(n - 1): # LU decomposition
        if abs(A[k, k]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k + 1:] = A[i, k + 1:] - factor * A[k, k + 1:]
            A[i, k] = factor

    y = np.zeros(n) # Forward substitution (Ly = b)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= A[i, j] * y[j]

    x = np.zeros(n) # Back substitution (Ux = y)
    x[n - 1] = y[n - 1] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]

    return x

for t in range(1, num_steps + 1, 1):
    for i in range(1, Nx):
        b[i] = T[i]

    b[0] = 0  # BC at x=0
    b[Nx] = (H * dx / alpha) * To  # BC at x=a

    T_new = solve_heat_equation(A, b)
    T = T_new

    if t * dt in time_intervals: # Store results at given time intervals
        T_profiles[t * dt] = T.copy()

    if t in time_steps:
        q_dot = np.zeros(Nx)
        for i in range(1, Nx - 1):
            q_dot[i] = -k * (T[i] - T[i - 1]) / dx  
        q_dot[0] = -k * (T[1] - T[0]) / dx  # Forward difference at the left boundary
        q_dot[Nx - 1] = -k * (T[Nx - 1] - T[Nx - 2]) / dx  # Backward difference at the right boundary
        flux_profiles[t * dt] = q_dot.copy()

plt.figure(figsize=(8, 6))
for time_val, q_dot in flux_profiles.items():
    plt.plot(x, q_dot, label=f't = {time_val}s')

end_time = tme.time()  # Use tme instead of time

plt.xlabel('Position x (m)')
plt.ylabel('Heat Flux q (W/m^2)')
plt.title('Heat Flux Profile at Different Times (implicit)')
plt.legend()
plt.grid()
plt.show()

print(f"Execution time: {end_time - start_time:.4f} seconds")
