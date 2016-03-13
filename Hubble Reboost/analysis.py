import pandas as pd
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from sv_from_coe import sv_from_coe
from scipy.integrate import odeint


# Constants
mu = 398600                           # Gravitational parameter (km**3/s**2)
RE = 6378                             # Earth's radius (km)
wE = np.array([0, 0, 7.2921159e-5])   # Earth's angular velocity (rad/s)

# Satellite data:
CD = 2.2                                # Drag codfficient
# m = 925                              # Mass (kg)
A = 2**2                              # Area (m**2)


def orbit(a=200, p=200, drag=False):
    ra = a + RE
    rp = p + RE
    e = (ra - rp) / (ra + rp)
    a = rp / (1 - e)
    p = a * (1 - e**2)
    h = (mu * p) ** (0.5)
    incl = 0.4969  # hst
    RA = 1.5199  # hst
    w = 2.8841  # hst
    TA = 0.5652  # hst

    coe0 = np.array([h, e, RA, incl, w, TA])

    # Obtain the initial state vector
    R0, V0 = sv_from_coe(coe0, mu)
    y0 = np.array([R0, V0]).flatten()

    t0, tf, nout = 0, 5 * 52 * 24 * 60 * 60, 1000000
    tspan = np.linspace(t0, tf, nout)
    y = odeint(rates, y0, tspan, args=(drag,))

    return y, tspan


def rates(f, t, drag=False):
    '''
    This function calculates the spacecraft acceleration from its position and
    velocity at time t.
    '''

    R = f[0:3]                        # Position vector (km/s)
    r = np.linalg.norm(R)             # Distance from earth's center (km)
    alt = r - RE                      # Altitude (km)
                                      # Air density from US Standard Model
                                      # (kf/m**3)
    rho = atmosphere(alt)
    V = f[3:6]                        # Velocity vector (km/s)
                                      # Velocity relative to the atmosphere
                                      # (km/s)
    Vrel = V - np.cross(wE, R)
    vrel = np.linalg.norm(Vrel)       # Speed relative to the atmosphere (km/s)
    uv = Vrel / vrel                  # Relative velocity unit vector
    ap = (-CD * A / m * rho *         # Acceleration due to drag (m/s**2)
          (1000 * vrel)**2 / 2 * uv)  # (converting units of vrel from km/s to m/s)
    a0 = -mu * R / r**3               # Gravitational ecceleration (km/s**2)
    if drag:
        a = a0 + ap / 1000
    else:
        a = a0

    dfdt = np.array([V, a]).flatten()
    return dfdt


def atmosphere(z):
    '''
    Calculates density for altitudes from sea level through 1000 km using
    exponential interpolation.
    '''

    data = np.array([
        [0, 8.4, 1.225, 1.225],
        [100, 5.9, 5.25E-7, 5.75E-7],
        [150, 25.5, 1.73E-9, 1.99E-9],
        [200, 37.5, 2.41E-10, 3.65E-10],
        [250, 44.8, 5.97E-11, 1.20E-10],
        [300, 50.3, 1.87E-11, 4.84E-11],
        [350, 54.8, 6.66E-12, 2.18E-11],
        [400, 58.2, 2.62E-12, 1.05E-11],
        [450, 61.3, 1.09E-12, 5.35E-12],
        [500, 64.5, 4.76E-13, 2.82E-12],
        [550, 68.7, 2.14E-13, 1.53E-12],
        [600, 74.8, 9.89E-14, 8.46E-13],
        [650, 84.4, 4.73E-14, 4.77E-13],
        [700, 99.3, 2.36E-14, 2.73E-13],
        [750, 121, 1.24E-14, 1.59E-13],
        [800, 151, 6.95E-15, 9.41E-14],
        [850, 188, 4.22E-15, 5.67E-14],
        [900, 226, 2.78E-15, 3.49E-14],
        [950, 263, 1.98E-15, 2.21E-14],
        [1000, 296, 1.49E-15, 1.43E-14],
        [1250, 408, 5.70E-16, 2.82E-15],
        [1500, 516, 2.79E-16, 1.16E-15],
        [2000, 829, 9.09E-17, 3.80E-16],
        [2500, 1220, 4.23E-17, 1.54E-16],
        [3000, 1590, 2.54E-17, 7.09E-17],
        [3500, 1900, 1.77E-17, 3.67E-17],
        [4000, 2180, 1.34E-17, 2.11E-17],
        [4500, 2430, 1.06E-17, 1.34E-17],
        [5000, 2690, 8.62E-18, 9.30E-18],
        [6000, 3200, 6.09E-18, 5.41E-18],
        [7000, 3750, 4.56E-18, 3.74E-18],
        [8000, 4340, 3.56E-18, 2.87E-18],
        [9000, 4970, 2.87E-18, 2.34E-18],
        [10000, 5630, 2.37E-18, 1.98E-18],
        [15000, 9600, 1.21E-18, 1.16E-18],
        [20000, 14600, 7.92E-19, 8.42E-19],
        [25000, 20700, 5.95E-19, 6.81E-19],
        [30000, 27800, 4.83E-19, 5.84E-19],
        [35000, 36000, 4.13E-19, 5.21E-19],
        [35786, 37300, 4.04E-19, 5.12E-19]])

    # Geometric altitudes (km):
    #h = [0, 25, 30, 40, 50, 60, 70,
    #     80, 90, 100, 110, 120, 130, 140,
    #     150, 180, 200, 250, 300, 350, 400,
    #     450, 500, 600, 700, 800, 900, 1000]
    h = data[:,0].tolist()

    # Corresponding densities (kg/m^3) from USSA76:
    #r = [1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5,
    #     1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9,
    #     2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12,
    #     1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15]
    r = data[:,2].tolist()

    # Scale heights (km):
    #H = [7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661,
    #     5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322,
    #     21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019,
    #     60.980, 65.654, 76.377, 100.587, 147.203, 208.020, 271.648]
    H = data[:,1].tolist()

    # Handle altitudes outside of the range:
    if z > h[-1]:
        z = h[-1]
    elif z < 0:
        z = 0

    # Determine the interpolation interval:
    for j in range(len(h)):
        if z >= h[j] and z < h[j + 1]:
            i = j

    if z == 1000:
        i = len(h)

    # Exponential interpolation:
    density = r[i] * np.exp(-(z - h[i]) / H[i])
    return density


def process_orbit(a, p, drag):
    y, t = orbit(a, p, drag)
    res = pd.DataFrame(y)
    res.columns = ['R1', 'R2', 'R3', 'V1', 'V2', 'V3']
    res['T'] = t
    res['Day'] = res['T'] / (60 * 60 * 24)
    res['R'] = res.apply(lambda x: np.linalg.norm(np.array(x[['R1', 'R2', 'R3']])), axis=1)
    res['V'] = res.apply(lambda x: np.linalg.norm(np.array(x[['V1', 'V2', 'V3']])), axis=1)
    res['Altitude'] = res['R'] - RE
    res['Drag'] = drag
    return res


#for a, p, m in [[200, 200, 996.1285767], [536, 536, 881.1438172], [616, 616, 637]]:
for a, p, m in [[536, 536, 881.1438172], [616, 616, 637]]:
#for a, p, m in [[200, 200, 996.1285767]]:
    print(a, p)
    drag = process_orbit(a, p, True)
    drag.to_csv(str(m) + '_' + str(a) + '_' + str(p) + '.csv')
    #drag.to_csv('{}.csv'.format(p))

f, ax = plt.subplots()
a200 = pd.read_csv('996.1285767_200_200.csv', index_col=0).query('Altitude > -100')
a536 = pd.read_csv('881.1438172_536_536.csv', index_col=0).query('Altitude > -100')
a616 = pd.read_csv('637_616_616.csv', index_col=0).query('Altitude > -100')

ax.plot(a200.Day, a200.Altitude, color='r', label='200 km')
ax.plot(a536.Day, a536.Altitude, color='b', label='536 km')
ax.plot(a616.Day, a616.Altitude, color='g', label='616 km')
plt.ylim(0, 625)
plt.show()
