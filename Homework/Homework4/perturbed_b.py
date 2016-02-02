import numpy as np
from numpy import pi
from scipy.optimize import minimize


e = 0.0002935
a = 6920  # km
i, RA, w, M = 0.4969, 1.5199, 2.8841, 5.7184  # rad
mu = 398600.4  # km^3/s^2
n = (mu/(a**3))**0.5
T = 2 * pi * (a**3/mu)**(0.5)
dt = np.linspace(0, T, 1000)

elements = np.zeros((len(dt), 6)) + np.array([e, a, i, RA, w, M])
elements[:, -1] += n * dt

h = 52519.6585  # km/s

#h, e, RA, i, w, TA
