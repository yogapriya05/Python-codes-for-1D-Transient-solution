import numpy as np
import matplotlib.pyplot as plt

# Given thermal properties of steel
k = 50  # Thermal conductivity (W/m.K)
rho = 7800  # Density (kg/m^3)
c_p = 500  # Specific heat capacity (J/kg.K)
alpha = 1.282051e-5  # Thermal diffusivity (m^2/s)

# Plate properties
a = 0.02  # Plate thickness (m)
T_inf = 50  # Ambient temperature (C)
Ti = 200  # Initial temperature (C)
h = 100  # Convective heat transfer coefficient (W/m^2.K)

# Define spatial and temporal domains
N = 50  # Number of terms in the series
x = np.linspace(0, a, 1000)  # Spatial domain
time_intervals = np.linspace(5, 50, 10)  # Time in seconds (5s, 10s, ..., 50s)

# Solve for lambda_n using a numerical approach
lambda_n = []
for n in range(1, N + 1):
    guess = (n * np.pi) / (2 * a)
    tolerance = 1e-6
    max_iterations = 100
    lambda_val = guess
    for _ in range(max_iterations):
        f_val = np.tan(lambda_val * a) - (lambda_val * k / h)
        df_val = a * (1 / np.cos(lambda_val * a)) ** 2 - k / h
        new_lambda_val = lambda_val - f_val / df_val
        if np.abs(new_lambda_val - lambda_val) < tolerance:
            break
        lambda_val = new_lambda_val
    lambda_n.append(lambda_val)

lambda_n = np.array(lambda_n)

# Compute the solution
plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(time_intervals)))

for j, t in enumerate(time_intervals):
    q_dot = k * T_inf * np.ones_like(x)  # Initialize with T_inf
    for n in range(N):
        # Compute coefficients using numerical integration
        coeff = 2 / a * np.trapz((Ti - T_inf) * np.sin(lambda_n[n] * x), x)
        term = -k * coeff * lambda_n[n] * np.exp(-(lambda_n[n] ** 2) * alpha * t) * np.sin(lambda_n[n] * x)
        q_dot -= term

    plt.plot(x, q_dot, color=colors[j], linewidth=1.5, label=f't = {t:.0f} sec')

# Labels and legend
plt.xlabel('x (m)')
plt.ylabel('q̇ (W/m²)')
plt.title('Heat Flux Profile Over Time (analytical)')
plt.legend()
plt.grid()
plt.show()
