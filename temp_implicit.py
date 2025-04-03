import numpy as np
import matplotlib.pyplot as plt
#from scipy.linalg import solve

alpha = 1.282051e-5
a = 0.02
Nx = 100 #This is total number of divisions required
dx = a / Nx
dt = 0.001
H = 2
Ti = 200
To = 50

time_intervals = np.arange(5, 51, 5)
T_profiles = {}

fo = alpha * dt / dx**2
x = np.linspace(0, a, Nx + 1) #Position definition

T = np.ones(Nx + 1) * Ti # Initialize temperature profile

# Coefficient matrix for implicit scheme (tdma)
A = np.zeros((Nx + 1, Nx + 1)) #here, first part is number of elements and second part is number of times such arrays
b = np.zeros((Nx + 1))

#constructing the matrix
for i in range(1, Nx):
    A[i, i - 1] = -fo
    A[i, i] = 1 + 2 * fo
    A[i, i + 1] = -fo
    
A[0, 0] = 1  # Neumann BC at x=0 (T_x = 0)
A[0, 1] = -1

A[Nx, Nx - 1] = -1  # Robin BC at x=a
A[Nx, Nx] = 1 + (H * dx / alpha)

def solve_heat_equation(A, b):
    n = len(b)
    # Make copies to avoid modifying original arrays
    A_copy = A.copy()
    b_copy = b.copy()

    for k in range(n - 1): # LU decomposition
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

for t in range(1, int(max(time_intervals) / dt) + 1):
    for i in range(1, Nx):
        b[i] = T[i]

    b[0] = 0  # BC at x=0
    b[Nx] = (H * dx / alpha) * To  # BC at x=a
    
    T_new = solve_heat_equation(A, b)
    T = T_new
    
    if t * dt in time_intervals: # Store results at given time intervals
        T_profiles[t * dt] = T.copy()

plt.figure(figsize=(8, 6))
for time, profile in T_profiles.items():
    plt.plot(x, profile, label=f't = {time}s')

plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Profile Over Time (Implicit)')
plt.legend()
plt.grid()
plt.show()
