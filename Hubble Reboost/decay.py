import pandas as pd
import numpy as np
from numpy import pi


def lifetime(H, M):
    A = 4     # Satellite area (m^2)
    F10 = 70  # Solar Radio Flux (SFU)
    Ap = 0    # Geomagnetic A index

    print("    TIME  HEIGHT  PERIOD  MEAN MOTION       DECAY")
    print("  (days)    (km)  (mins)    (rev/day) (rev/day^2)")

    Re = 6378000
    Me = 5.98E+24   # Earth radius and mass (all SI units)
    G = 6.67E-11    # Universal constant of gravitation
    T, dT = 0., .1  # Time and time increment are in days
    D9 = dT * 3600 * 24  # Put time increment into seconds
    H1 = 1          # Print height increment
    H2 = H          # Print height
    R = Re + H * 1000  # R is orbital radius in metres
    P = 2 * pi * np.sqrt(R ** 3 / Me / G)  # P is period in seconds

    while True:
        SH = (900 + 2.5 * (F10 - 70) + 1.5 * Ap) / (27 - .012 * (H - 200))
        DN = 6E-10 * np.exp(-(H - 175) / SH)  # Atmospheric density
        dP = 3 * pi * A / M * R * DN * D9     # Decrement in orbital period
        if H <= H2:                           # Test for print
            Pm = P / 60
            MM = 1440 / Pm
            Decay = dP / dT / P * MM          # rev/day/day
            print("{:8.1f} {:7.1f} {:7.1f} {:12.4f} {:10.2e}".format(T, H, P / 60, MM, Decay))  # do print
            H2 = H2 - H1                      # Decrement print height
        if H < 0:
            print("Re-entry after {:3.0f} days ( {:4.2f} years)".format(T, T / 365))
            return T
        P = P - dP
        T = T + dT
        R = (G * Me * P * P / 4 / pi / pi) ** (1/3)
        H = (R - Re) / 1000

res = []
for H in np.linspace(100, 650, 5):
    for M in np.linspace(1000, 600, 5):
        T = lifetime(H, M)
        print(H, M, T)
        res.append([H, M, T])

res = pd.DataFrame(res)
res.columns = ['H', 'M', 'T']
