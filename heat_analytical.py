import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad

alpha = 1.282051e-5 
k = 50 
rho = 7800  
c_p = 500  
a = 0.02  
T_inf = 50  
Ti = 200 
h = 100 
N = 50  # Number of terms in the series
x = np.linspace(0, a, 1000)  
time_intervals = np.linspace(5, 50, 10) 

def lambda_eq(lambda_n):
    return np.tan(lambda_n * a) - (lambda_n * k / h)

lambda_n = np.array([fsolve(lambda_eq, (n * np.pi) / (2 * a))[0] for n in range(1, N + 1)])

plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(time_intervals)))

for j, t in enumerate(time_intervals):
    q_dot = k * T_inf * np.ones_like(x) 
    for n in range(N):
        coeff, _ = quad(lambda x: (Ti - T_inf) * np.sin(lambda_n[n] * x), 0, a)
        coeff *= 2 / a
        term = -k * coeff * lambda_n[n] * np.exp(-(lambda_n[n] ** 2) * alpha * t) * np.sin(lambda_n[n] * x)
        q_dot -= term

    plt.plot(x, q_dot, color=colors[j], linewidth=1.5, label=f't = {t:.0f} sec')

plt.xlabel('x (m)')
plt.ylabel('q̇ (W/m²)')
plt.title('Heat Flux Profile Over Time (analytical)')
plt.legend()
plt.grid()
plt.show()
