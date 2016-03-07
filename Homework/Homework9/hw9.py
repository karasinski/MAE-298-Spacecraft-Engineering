import numpy as np

###############################################################################
# Problem 1
# Part 1
a = 10 * 1/100  # m
mu = 1/1000 * (100)**3  # kg/m^3
A = a**2  # m^2
CD = 2.2
theta = np.deg2rad(30)  # rad

n1 = np.array([np.cos(theta), np.sin(theta), 0])  # red face
n2 = np.array([np.sin(theta), np.cos(theta), 0])  # top face
V = np.array([1, 0, 0])

low = {'v': 7789,
       'rho': 8.28E-10,
       'alt': '200km'}
high = {'v': 7702,
        'rho': 8.05E-11,
        'alt': '350km'}

m1 = mu * (a * a * (a/2))
m2 = (mu/2) * (a * a * (a/2))
r1 = np.array([0, a/4, 0])
r2 = np.array([0, -a/4, 0])
Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta),  np.cos(theta), 0],
               [            0,              0, 1]])
r1_rot = np.dot(r1, Rz)
r2_rot = np.dot(r2, Rz)

x = (m1 * r1_rot[0] + m2 * r2_rot[0])/(m1 + m2)
y = (m1 * r1_rot[1] + m2 * r2_rot[1])/(m1 + m2)
r = np.array([x, y, 0])

print('\nDrag Torque')
for alt in [low, high]:
    T = np.array([0., 0., 0.])
    for n in [n1, n2]:
        F = 0.5 * alt['rho'] * alt['v']**2 * CD * np.dot(n, V) * A * (-V)
        T += np.cross(r, F)
    print(F, T, 'at altitude {}'.format(alt['alt']))

###############################################################################
# Part 2
P = 4.67E-6
s = np.array([1, 0, 0])
a_i = -P * A * (1 - .21) * np.cos(theta)
b_i = -2 * P * A * (.21 * np.cos(theta) + (1/3) * 0.1) * np.cos(theta)

print('\nSolar Pressure Torque')
for alt in [low, high]:
    T = np.array([0., 0., 0.])
    for n in [n1, n2]:
        F = a_i * s + b_i * n
        T += np.cross(r, F)
    print(F, T, 'at altitude {}'.format(alt['alt']))

###############################################################################
# Part 3
mu = 0.3986E15
r_earth = np.array([0., 1., 0.])

print('\nGravity Gradient Torque')
for alt in [low, high]:
    T = np.array([0., 0., 0.])
    for m, r_rot in zip([m1, m2], [r1_rot, r2_rot]):
        rr = r_rot + np.array([0, -6371E3, 0])
        F = ((mu * m) / np.linalg.norm(rr)**2) * (-r_earth)
        T += np.cross(r, F)
    print(F, T, 'at altitude {}'.format(alt['alt']))

###############################################################################
# Part 4
n = 25
I = 0.1
A = 25 * np.pi * (1/100)**2
c = np.array([np.cos(theta), np.sin(theta), 0])
B = np.array([0., 0., 1.])

print('\nMagnetic Dipole Torque')
for alt in [low, high]:
    T = np.array([0., 0., 0.])
    for n in [n1, n2]:
        M = n * I * A * c
        T += np.cross(M, B)
    print(F, T, 'at altitude {}'.format(alt['alt']))

###############################################################################
