import numpy as np
import matplotlib.pyplot as plt


a = 0.02  
k = 50  
H = 2  # h/k
alpha = 1.282051e-5 
Ti = 200 
To = 50  
Nx = 100  
dx = a / Nx  
dt = 0.001 

time_intervals = np.arange(5, 55, 5)  
num_steps = int(max(time_intervals) // dt) 

T = np.ones(Nx) * Ti

flux_profiles = {}
time_steps = [int(t / dt) for t in time_intervals]

for n in range(1, num_steps + 1, 1):
    T_new = T
    for i in range(1, Nx - 1, 1):
        T_new[i] = T[i] + Fo * (T[i + 1] - 2 * T[i] + T[i - 1])

    T_new[0] = T_new[1] 
    T_new[Nx - 1] = (T[Nx - 2] + (H * dx / alpha) * To) / (1 + H * dx / alpha)  

    T = T_new
    
    if n in time_steps: # Store heat flux profile at required time steps
        q_dot = np.zeros(Nx)
        for i in range(1, Nx - 1):
            q_dot[i] = -k * (T[i + 1] - T[i - 1]) / (2 * dx)
        q_dot[0] = -k * (T[1] - T[0]) / dx  # Forward difference at the left boundary
        q_dot[Nx - 1] = -k * (T[Nx - 1] - T[Nx - 2]) / dx  # Backward difference at the right boundary
        flux_profiles[n * dt] = q_dot.copy()

plt.figure(figsize=(8, 6))
x = np.linspace(0, a, Nx)
for time, q_dot in flux_profiles.items():
    plt.plot(x, q_dot, label=f't = {time:.1f} s')

plt.xlabel('Position (m)')
plt.ylabel('Heat Flux (W/m²)')
plt.title('Heat Flux Profile Over Time (explicit)')
plt.legend()
plt.grid()
plt.show()
