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
x = np.linspace(0, a, 1000)
time_intvl = np.linspace(5, 50, 10)

def lambda_eq(lambda_n):
    return np.tan(lambda_n * a) - (lambda_n * k / h)

lambda_n = np.array([fsolve(lambda_eq, (n * np.pi) / (2 * a))[0] for n in range(1, N + 1)])

plt.figure()
colors = plt.cm.jet(np.linspace(0, 1, len(time_intvl)))

for j, t in enumerate(time_intvl):
    T = np.ones_like(x) * T_inf 

    for n in range(N):
        integrand = lambda x: (Ti - T_inf) * np.sin(lambda_n[n] * x)
        coeff, _ = quad(integrand, 0, a)
        coeff *= (2 / a)
        T += coeff * np.exp(-(lambda_n[n]) ** 2 * alpha * t) * np.cos(lambda_n[n] * x)

    plt.plot(x, T, color=colors[j], linewidth=1.5, label=f't = {t} sec')

plt.xlabel('x (m)')
plt.ylabel('T(x,t) (°C)')
plt.title('Temperature Profile Over Time (analytical)')
plt.legend()
plt.grid()
plt.show()
