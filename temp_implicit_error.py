import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.linalg import solve

k = 50  
rho = 7800  
c_p = 500  
alpha = 1.282051e-5  
a = 0.02  
T_inf = 50  
Ti = 200 
h = 100 
Nx = 100  
dx = a / Nx
dt = 0.001 
H = 2 # h/k
To = 50  
N = 50  # Number of terms in the analytical series
x = np.linspace(0, a, Nx + 1)
time_intervals = np.arange(5, 51, 5)

def lambda_eq(lambda_n):
    return np.tan(lambda_n * a) - (lambda_n * k / h)

lambda_n = np.array([fsolve(lambda_eq, (n * np.pi) / (2 * a))[0] for n in range(1, N + 1)])

def analytical_solution(x, t): # Analytical solution function
    T = np.ones_like(x) * T_inf
    for n in range(N):
        integrand = lambda x: (Ti - T_inf) * np.sin(lambda_n[n] * x)
        coeff, _ = quad(integrand, 0, a)
        coeff *= (2 / a)
        T += coeff * np.exp(-(lambda_n[n]) ** 2 * alpha * t) * np.cos(lambda_n[n] * x)
    return T

# Implicit scheme setup
fo = alpha * dt / dx**2
T = np.ones(Nx + 1) * Ti
A = np.zeros((Nx + 1, Nx + 1))
b = np.zeros((Nx + 1))

for i in range(1, Nx):
    A[i, i - 1] = -fo
    A[i, i] = 1 + 2 * fo
    A[i, i + 1] = -fo

A[0, 0] = 1 
A[0, 1] = -1
A[Nx, Nx - 1] = -1  
A[Nx, Nx] = 1 + (H * dx / alpha)

T_profiles = {}
for t in range(1, int(max(time_intervals) / dt) + 1):
    for i in range(1, Nx):
        b[i] = T[i]
    b[0] = 0  # BC at x=0
    b[Nx] = (H * dx / alpha) * To  # BC at x=a
    T_new = solve(A, b)
    T = T_new
    if t * dt in time_intervals:
        T_profiles[t * dt] = T.copy()

plt.figure(figsize=(8, 6)) # Compute error and plot
for t in time_intervals:
    T_analytical = analytical_solution(x, t)
    T_numerical = T_profiles[t]
    error = np.abs(T_analytical - T_numerical)
    plt.plot(x, error, label=f't = {t}s')

plt.xlabel('Position (m)')
plt.ylabel('Absolute Error (°C)')
plt.title('Error Between Analytical and Numerical (Implicit) Solution')
plt.legend()
plt.grid()
plt.show()
