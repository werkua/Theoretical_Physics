# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 01:17:00 2024

@author: Werka
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({'font.size': 14})

# Symulacja parametrów
n = 6
nt = 5000
q = 1.0
B = 1.0
m = 1.0
wc = q * B / m  # okres obiegu
T = 2 * np.pi / wc
dt = 5 * T / nt  # krok czasowy


def f0(t, s):
    return s[3]/m


def f1(t, s):
    return s[4] / (s[0] ** 2) - 0.5*q*B/m


def f2(t, s):
    return s[5]/m


def f3(t, s):
    return s[4] ** 2 / ( m* s[0] ** 3) - s[0] / 4

def f4(t, s):
    return 0


def f5(t, s):
    return 0


def pochodne(t, s):
    k = np.zeros(6)
    k[0] = f0(t, s)
    k[1] = f1(t, s)
    k[2] = f2(t, s)
    k[3] = f3(t, s)
    k[4] = f4(t, s)
    k[5] = f5(t, s)
    return k


def rk4_vec(t, s, dt):
    k1 = dt * pochodne(t, s)
    k2 = dt * pochodne(t + dt / 2, s + 0.5 * k1)
    k3 = dt * pochodne(t + dt / 2, s + 0.5 * k2)
    k4 = dt * pochodne(t + dt, s + k3)
    return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0




# Warunki początkowe r0, phi0, z0, pr0, pphi0, pz0
conditions = np.array([
    [1.5, 1.25 * np.pi, 0, 0, q * B * (1.5 ** 2) / 2, 0],
    [1, 0, 0, 0, -q * B * (1 ** 2) / 2, 0],
    [2, 0, 0, 0, -q * B * (2 ** 2) / 2, 0],
    [2, 0, 0, 2, -q * B * (2 ** 2) / 2, 0]
])

plt.figure(figsize=(8, 5))
# Symulacja trajektorii
for i, cond in enumerate(conditions):
    time = []
    r = np.zeros(nt)
    phi = np.zeros(nt)
    z = np.zeros(nt)
    pr = np.zeros(nt)
    pphi = np.zeros(nt)
    pz = np.zeros(nt)
    s = cond
    
    # wsp kartezjanskie
    x = np.zeros(nt)
    y = np.zeros(nt)
    for j in range(nt):
        t = j * dt
        time.append(t)
        r[j], phi[j], z[j], pr[j], pphi[j], pz[j] = s[0], s[1], s[2], s[3], s[4], s[5]
        x[j] = r[j] * np.cos(phi[j])
        y[j] = r[j] * np.sin(phi[j])
        s = rk4_vec(t, s, dt)
    if(i==0):
        plt.plot(time, r, label=f'wp {i}', color ='red')
    else:
        plt.plot(time, r, label=f'wp {i}')

# Ustawienia wykresu
plt.ylabel('r')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.xlim(0)
plt.savefig('r_t')
#plt.show()


plt.figure(figsize=(10, 11))


# Symulacja trajektorii
for i, cond in enumerate(conditions):
    time = []
    r = np.zeros(nt)
    phi = np.zeros(nt)
    z = np.zeros(nt)
    pr = np.zeros(nt)
    pphi = np.zeros(nt)
    pz = np.zeros(nt)
    s = cond
    # wsp kartezjanskie
    x = np.zeros(nt)
    y = np.zeros(nt)
    for j in range(nt):
        t = j * dt
        time.append(t)
        r[j], phi[j], z[j], pr[j], pphi[j], pz[j] = s[0], s[1], s[2], s[3], s[4], s[5]
        x[j] = r[j] * np.cos(phi[j])
        y[j] = r[j] * np.sin(phi[j])
        s = rk4_vec(t, s, dt)
    if(i==0):
        plt.scatter(x, y, label=f'wp {i}', color ='red')
    else:
        plt.plot(x, y, label=f'wp {i}')

# Ustawienia wykresu

plt.ylabel('y(t)')
plt.xlabel('x(t)')
plt.legend()
plt.grid()
plt.savefig('y_x')
#plt.show()

plt.figure(figsize=(8, 5))
# Symulacja trajektorii
for i, cond in enumerate(conditions):
    time = []
    r = np.zeros(nt)
    phi = np.zeros(nt)
    z = np.zeros(nt)
    pr = np.zeros(nt)
    pphi = np.zeros(nt)
    pz = np.zeros(nt)
    s = cond
    # wsp kartezjanskie
    x = np.zeros(nt)
    y = np.zeros(nt)
    for j in range(nt):
        t = j * dt
        time.append(t)
        r[j], phi[j], z[j], pr[j], pphi[j], pz[j] = s[0], s[1], s[2], s[3], s[4], s[5]
        x[j] = r[j] * np.cos(phi[j])
        y[j] = r[j] * np.sin(phi[j])
        s = rk4_vec(t, s, dt)
    if(i==0):
        plt.scatter(time, phi, label=f'wp {i}', color ='red')
    else:
        plt.plot(time, phi, label=f'wp {i}')

# Ustawienia wykresu
plt.ylabel('φ(t)')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.savefig('phi_t')
#plt.show()

plt.figure(figsize=(8, 5))
# Symulacja trajektorii
for i, cond in enumerate(conditions):
    time = []
    r = np.zeros(nt)
    phi = np.zeros(nt)
    z = np.zeros(nt)
    pr = np.zeros(nt)
    pphi = np.zeros(nt)
    pz = np.zeros(nt)
    s = cond
    # wsp kartezjanskie
    x = np.zeros(nt)
    y = np.zeros(nt)
    for j in range(nt):
        t = j * dt
        time.append(t)
        r[j], phi[j], z[j], pr[j], pphi[j], pz[j] = s[0], s[1], s[2], s[3], s[4], s[5]
        x[j] = r[j] * np.cos(phi[j])
        y[j] = r[j] * np.sin(phi[j])
        s = rk4_vec(t, s, dt)
    if(i==0):
        plt.scatter(time, pr, label=f'wp {i}', color ='red')
    else:
        plt.plot(time, pr, label=f'wp {i}')

# Ustawienia wykresu
plt.ylabel('$p_{r}(t)$')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.savefig('pr_t')
#plt.show()



#  ==================== energia ===================

plt.figure(figsize=(8, 5))
# Symulacja trajektorii
for i, cond in enumerate(conditions):
    time = []
    r = np.zeros(nt)
    phi = np.zeros(nt)
    z = np.zeros(nt)
    pr = np.zeros(nt)
    pphi = np.zeros(nt)
    pz = np.zeros(nt)
    s = cond
    # wsp kartezjanskie
    x = np.zeros(nt)
    y = np.zeros(nt)
    E = np.zeros(nt)

    for j in range(nt):
        t = j * dt
        time.append(t)
        r[j], phi[j], z[j], pr[j], pphi[j], pz[j] = s[0], s[1], s[2], s[3], s[4], s[5]
        x[j] = r[j] * np.cos(phi[j])
        y[j] = r[j] * np.sin(phi[j])
        E[j] = 1/(2*m)*(pr[j]**2+ pphi[j]**2/r[j]**2+pz[j]**2) - q*B/(2*m) * pphi[j] + (q*B*r[j])**2/(8*m)
        s = rk4_vec(t, s, dt)
    if(i==0):
        plt.scatter(time, E, label=f'wp {i}', color ='red')
    else:
        plt.plot(time, E, label=f'wp {i}')

# Ustawienia wykresu
plt.ylabel('$E(t)$')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.savefig('E_t')
plt.show()


