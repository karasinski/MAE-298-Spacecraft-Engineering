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
CD = 2.5                              # Drag codfficient
m = 11110                             # Mass (kg)
A = 55.44                             # Area (m**2)


def orbit(drag):
    coe0 = np.array([52519.6585, 0.0002935, 1.5199, 0.4969, 2.8841, 0.5652])

    # Obtain the initial state vector
    R0, V0 = sv_from_coe(coe0, mu)
    y0 = np.array([R0, V0]).flatten()

    t0, tf, nout = 0, 5728.8, 10000
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
    rho = atmosphere(alt)             # Air density from US Standard Model (kf/m**3)
    V = f[3:6]                        # Velocity vector (km/s)
    Vrel = V - np.cross(wE, R)        # Velocity relative to the atmosphere (km/s)
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

    # Geometric altitudes (km):
    h = (
        [0, 25, 30, 40, 50, 60, 70,
         80, 90, 100, 110, 120, 130, 140,
         150, 180, 200, 250, 300, 350, 400,
         450, 500, 600, 700, 800, 900, 1000])

    # Corresponding densities (kg/m^3) from USSA76:
    r = (
        [1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5,
         1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9,
         2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12,
         1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15])

    # Scale heights (km):
    H = (
        [7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661,
         5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322,
         21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019,
         60.980, 65.654, 76.377, 100.587, 147.203, 208.020, 270.010])

    # Handle altitudes outside of the range:
    if z > 1000:
        z = 1000
    elif z < 0:
        z = 0

    # Determine the interpolation interval:
    for j in range(len(h) - 1):
        if z >= h[j] and z < h[j + 1]:
            i = j
    if z == 1000:
        i = len(h) - 1

    # Exponential interpolation:
    density = r[i] * np.exp(-(z - h[i]) / H[i])
    return density


def process_orbit(drag):
    y, t = orbit(drag)
    res = pd.DataFrame(y)
    res.columns = ['R1', 'R2', 'R3', 'V1', 'V2', 'V3']
    res['T'] = t
    res['R'] = res.apply(lambda x: np.linalg.norm(np.array(x[['R1', 'R2', 'R3']])), axis=1)
    res['V'] = res.apply(lambda x: np.linalg.norm(np.array(x[['V1', 'V2', 'V3']])), axis=1)
    res['Drag'] = drag
    return res


drag = process_orbit(True)
no_drag = process_orbit(False)
res = pd.concat((drag, no_drag))
diff = res.query('Drag') - res.query('not Drag')
diff['T'] = res.query('Drag')['T']

f, ax = plt.subplots(2, sharex=True)
ax[0].set_ylabel('r, km')
ax[1].set_ylabel('v, km/s')
diff.plot(x='T', y='R', ax=ax[0], legend=False)
diff.plot(x='T', y='V', ax=ax[1], legend=False)
ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
#plt.savefig('p7.pdf')
plt.show()
