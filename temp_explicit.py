import numpy as np
import matplotlib.pyplot as plt

alpha = 1.282051e-5
a = 0.02
Nx = 100  
dx = a / Nx  
dt = 0.0001 
H = 2  # h/k
Ti = 200
To = 50
time_intervals = np.arange(5, 56, 5)

Fo = alpha * dt / dx ** 2
if Fo > 0.5:
    raise ValueError(f"Stability condition violated! Fo = {Fo}. Reduce dt or increase Nx.")

num_steps = int(max(time_intervals) // dt)  # Total number of time steps
T_profiles = {}

T = np.ones(Nx + 1) * Ti # Initialize temperature field
x = np.linspace(0, a, Nx + 1)

for n in range(1, num_steps + 1, 1):
    T_new = T

    for i in range(1, Nx):
        T_new[i] = T[i] + Fo * (T[i + 1] - 2 * T[i] + T[i - 1])

    T_new[0] = T[1]
    T_new[Nx] = (T[Nx - 1] + (H * dx / alpha) * To) / (1 + (H * dx / alpha))

    T = T_new

    if (n * dt) in time_intervals:  # Store profiles at required time intervals
        T_profiles[n * dt] = T.copy()

plt.figure(figsize=(8, 6))
for time, profile in T_profiles.items():
    plt.plot(x, profile, label=f'Time = {t}s')

plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Profile Over Time (Explicit)')
plt.legend()
plt.grid()
plt.show()
