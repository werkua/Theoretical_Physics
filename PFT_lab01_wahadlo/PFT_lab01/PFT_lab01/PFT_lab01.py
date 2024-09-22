import numpy as np
import matplotlib.pyplot as plt
from funkcje import pochodne, rk4_vec, energia_kinetyczna, energia_potencjalna, energia_calkowita, rozw_analityczne, wykres_trajektorii

# Warunki początkowe
initial_conditions_list = [(np.radians(angle), 0.0) for angle in [4, 45, 90, 135, 175]]

# Czas symulacji
dt = 0.01
nt = 1000
t = np.linspace(0, dt * (nt - 1), nt)

# Rozwiązania numeryczne dla różnych warunków początkowych
solutions = []
solutions_a = []
for initial_conditions in initial_conditions_list:
    solution_numeric = rk4_vec(pochodne, t, initial_conditions)
    solution_analytic  = rozw_analityczne(t, initial_conditions)
    solutions.append(solution_numeric)
    solutions_a.append(solution_analytic)


# Wykresy dla różnych warunków początkowych
plt.figure(figsize=(12, 8))

# Różnica analityczna vs. numeryczna
for (solution, solution_a) in (solutions, solutions_a):
    plt.plot(t, solution[:, 0] - solution_a[:, 0])
plt.title('Różnice ')
plt.xlabel('Czas (s)')
plt.ylabel('Kąt (rad)')
plt.grid(True)
plt.savefig("comparison_diff")


# Wykresy dla różnych warunków początkowych
plt.figure(figsize=(12, 8))

# Wykresy φ(t)
plt.subplot(2, 2, 1)
for solution in solutions:
    plt.plot(t, solution[:, 0])
plt.title('Kąt (φ) w czasie')
plt.xlabel('Czas (s)')
plt.ylabel('Kąt (rad)')
plt.grid(True)

# Wykresy energii kinetycznej T(t)
plt.subplot(2, 2, 2)
for solution in solutions:
    plt.plot(t, energia_kinetyczna(solution))
plt.xlabel('Czas (s)')
plt.ylabel('Energia kinetyczna (J)')
plt.grid(True)

# Wykresy energii potencjalnej U(t)
plt.subplot(2, 2, 3)
for solution in solutions:
    plt.plot(t, energia_potencjalna(solution))
plt.xlabel('Czas (s)')
plt.ylabel('Energia potencjalna (J)')
plt.grid(True)

# Wykresy energii całkowitej E(t)
plt.subplot(2, 2, 4)
for solution in solutions:
    T = energia_kinetyczna(solution)
    U = energia_potencjalna(solution)
    E = energia_calkowita(T, U)
    plt.plot(t, E)
plt.xlabel('Czas (s)')
plt.ylabel('Energia całkowita (J)')
plt.grid(True)

plt.tight_layout()
plt.savefig("wykres_energia")
plt.show()

# Wykres trajektorii wahadła w przestrzeni konfiguracyjnej
wykres_trajektorii(solutions)


# Wykres porównawzy rozw analitycznego i numerycznego
s0_initial = np.radians(4)  # Przykładowa wartość początkowa kąta
s1_initial = 0.0  # Przykładowa wartość początkowa prędkości kątowej
initial_conditions = np.array([s0_initial, s1_initial])


solution_numeric = rk4_vec(pochodne, t, initial_conditions)

# Rozwiązanie analityczne
solution_analytic = rozw_analityczne(t, s0_initial)

# Wykres porównawczy
plt.figure(figsize=(10, 6))
plt.plot(t, solution_numeric[:, 0], label='Rozwiązanie numeryczne')
plt.plot(t, solution_analytic, label='Rozwiązanie analityczne', linestyle='--')
plt.xlabel('Czas (s)')
plt.ylabel('Kąt (rad)')
plt.legend()
plt.grid(True)
plt.savefig("wykres_comparison_4")
plt.show()



# Wykres porównawzy rozw analitycznego i numerycznego
s0_initial = np.radians(175)  # Przykładowa wartość początkowa kąta
s1_initial = 0.0  # Przykładowa wartość początkowa prędkości kątowej
initial_conditions = np.array([s0_initial, s1_initial])


solution_numeric = rk4_vec(pochodne, t, initial_conditions)

# Rozwiązanie analityczne
solution_analytic = rozw_analityczne(t, s0_initial)

# Wykres porównawczy
plt.figure(figsize=(10, 6))
plt.plot(t, solution_numeric[:, 0], label='Rozwiązanie numeryczne')
plt.plot(t, solution_analytic, label='Rozwiązanie analityczne', linestyle='--')
plt.xlabel('Czas (s)')
plt.ylabel('Kąt (rad)')
plt.legend()
plt.grid(True)
plt.savefig("wykres_comparison_175")
plt.show()




# Zakres maksymalnego wychylenia
max_wychylenie_range = np.linspace(0, np.pi, 100)  # Od 0 do π radianów (0 do 180 stopni)

# Obliczenie okresu dla każdego maksymalnego wychylenia
okresy = okres_wahadla(max_wychylenie_range)

# Wykres okresu wahadła w funkcji maksymalnego wychylenia
plt.figure(figsize=(8, 6))
plt.plot(max_wychylenie_range, okresy)
plt.title('Okres wahadła w funkcji maksymalnego wychylenia')
plt.xlabel('Maksymalne wychylenie (rad)')
plt.ylabel('Okres wahadła (s)')
plt.grid(True)
plt.show()