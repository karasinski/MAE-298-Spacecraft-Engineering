import pandas as pd
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from sv_from_coe import sv_from_coe
from scipy.integrate import odeint


# Constants
mu = 398600                 # Gravitational parameter (km**3/s**2)
RE = 6378                   # Earth's radius (km)
wE = np.array([0, 0, 7.2921159e-5])   # Earth's angular velocity (rad/s)

# Satellite data:
CD = 2.5                    # Drag codfficient
m = 11110                    # Mass (kg)
A = 55.44             # Frontal area (m**2)


def orbit(drag):
    # Conversion factors:
    hours = 3600                   # Hours to seconds
    days = 24 * hours               # Days to seconds
    years = 365 * days             # Years to seconds
    deg = pi / 180                 # Degrees to radians

    ## Initial orbital parameters (given):
    #rp = RE + 215               # perigee radius (km)
    #ra = RE + 939               # apogee radius (km)
    #RA = 339.94 * deg             # Right ascencion of the node (radians)
    #i = 65.1 * deg               # Inclination (radians)
    #w = 58 * deg                 # Argument of perigee (radians)
    #TA = 332 * deg                # True anomaly (radians)

    ## Initial orbital parameters (inferred):
    #e = (ra - rp) / (ra + rp)        # eccentricity
    #a = (rp + ra) / 2            # Semimajor axis (km)
    #h = np.sqrt(mu * a * (1 - e**2))     # angular momentrum (km**2/s)
    #T = 2 * pi / np.sqrt(mu) * a**1.5    # Period (s)

    # Store initial orbital elements (from above) in the vector coe0:
    #coe0 = np.array([h, e, RA, i, w, TA])
    coe0 = np.array([52519.6585, 0.0002935, 1.5199, 0.4969, 2.8841, 0.5652])

    # Obtain the initial state vector from Algorithm 4.5 (sv_from_coe):
    R0, V0 = sv_from_coe(coe0, mu)  # R0 is the initial position vector
                                    # V0 is the initial velocity vector
    # Magnitudes of R0 and V0
    r0, v0 = np.linalg.norm(R0), np.linalg.norm(V0)

    # Use ODE45 to integrate the equations of motion d/dt(R,V) = f(R,V)
    # from t0 to tf:
    t0, tf = 0, 5728.8       # Initial and final times (s)
    y0 = np.array([R0, V0]).flatten()               # Initial state vector
    nout = 10000                  # Number of solution points to output
    tspan = np.linspace(t0, tf, nout)  # Integration time interval
    y = odeint(rates, y0, tspan, args=(drag,)) # y is the state vector history

    return y, tspan


def rates(f, t, drag=False):
    '''
    This function calculates the spacecraft acceleration from its position and
    velocity at time t.
    '''

    R = f[0:3]            # Position vector (km/s)
    r = np.linalg.norm(R)            # Distance from earth's center (km)
    alt = r - RE             # Altitude (km)
    rho = atmosphere(alt)    # Air density from US Standard Model (kf/m**3)
    V = f[3:6]            # Velocity vector (km/s)
    Vrel = V - np.cross(wE, R)    # Velocity relative to the atmosphere (km/s)
    # Speed relative to the atmosphere (km/s)
    vrel = np.linalg.norm(Vrel)
    uv = Vrel / vrel          # Relative velocity unit vector
    ap = (-CD * A / m * rho *       # Acceleration due to drag (m/s**2)
          (1000 * vrel)**2 / 2 * uv)  # (converting units of vrel from km/s to m/s)
    a0 = -mu * R / r**3          # Gravitational ecceleration (km/s**2)
    if drag:
        a = a0 + ap / 1000       # Total acceleration (km/s**2)
    else:
        a = a0
    # Velocity and the acceleraion returned to ode45
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

f, ax = plt.subplots()
diff.plot(x='T', y='R', ax=ax)
diff.plot(x='T', y='V', secondary_y=True, ax=ax)
plt.tight_layout()
plt.savefig('p7.pdf')
#plt.show()
