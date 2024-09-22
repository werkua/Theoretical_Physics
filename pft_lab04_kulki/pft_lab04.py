import numpy as np
import matplotlib.pyplot as plt

# Parametry symulacji
N = 50
dt = 0.02
delta = 0.1
alpha = 1
m = 1
nt = 5000
sigma = 3 * delta

# Funkcja obliczająca pochodne w procedurze RK4
def calculate_derivatives(s, s_dot, N, delta, alpha, m):
    s_dot_new = np.zeros_like(s_dot)
    s_double_dot = np.zeros_like(s_dot)
    
    # Obliczenie nowych prędkości
    for i in range(1, N):
        s_double_dot[i] = (alpha / m) * (s[i-1] - 2 * s[i] + s[i+1])
    
    # Warunki brzegowe
    s_double_dot[0] = 0
    s_double_dot[N] = 0
    
    # Nowe prędkości
    s_dot_new[1:N] = s_double_dot[1:N]
    
    return s_dot_new

# Warunki początkowe
x_max = delta * N
x_eq = np.arange(0, x_max + delta, delta) 
s_0 = x_eq + delta / 3 * np.exp(-((x_eq - x_max / 2)**2) / (2 * sigma**2)) # (20)
s_dot_0 = np.zeros_like(s_0) # (21)

# Lista przechowująca energię kinetyczną, potencjalną i całkowitą w kolejnych krokach czasowych
kinetic_energy = []
potential_energy = []
total_energy = []


particle_displacements = [[] for _ in range(N + 1)]


s = s_0.copy()
s_dot = s_dot_0.copy()

for t in range(nt):
    # RK4
    k1 = calculate_derivatives(s, s_dot, N, delta, alpha, m)
    k2 = calculate_derivatives(s + 0.5 * dt * k1, s_dot, N, delta, alpha, m)
    k3 = calculate_derivatives(s + 0.5 * dt * k2, s_dot, N, delta, alpha, m)
    k4 = calculate_derivatives(s + dt * k3, s_dot, N, delta, alpha, m)
    
    s_dot += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    # nowe polozenia
    s += dt * s_dot
    
    # Zapis wychyleń cząstek
    for i in range(N + 1):
        particle_displacements[i].append(s[i])
    
    # Obliczenie energii kinetycznej
    kinetic = 0.5 * m * np.sum(s_dot[1:N]**2)
    kinetic_energy.append(kinetic)
    
    # Obliczenie energii potencjalnej
    potential = 0
    for i in range(1, N+1):
        potential += 0.5 * alpha * (s[i-1] - s[i] + delta)**2
    potential_energy.append(potential)
    
    # Całkowita energia
    total_energy.append(kinetic + potential)

kinetic_energy = np.array(kinetic_energy)
potential_energy = np.array(potential_energy)
total_energy = np.array(total_energy)

# Czas symulacji
time = np.arange(0, nt*dt, dt)

plt.figure(figsize=(15, 6))
plt.plot(time, kinetic_energy, label='Energia kinetyczna')
plt.plot(time, potential_energy, label='Energia potencjalna')
plt.plot(time, total_energy, label='Całkowita energia')
plt.xlabel('Czas')
plt.ylabel('Energia')
plt.title('Energia w funkcji czasu')
plt.legend()
plt.grid(True)
plt.show()