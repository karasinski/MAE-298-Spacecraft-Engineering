import numpy as np
from numpy import sin, cos
import pandas as pd
from scipy.special import j1
import matplotlib.pyplot as plt


def problem1():
    def hohmann(r1, r2, mu=398600.4, re=6371):
        r1 += re
        r2 += re

        v1 = abs(np.sqrt(mu / r1) * (np.sqrt((2 * r2) / (r1 + r2)) - 1))
        v2 = abs(np.sqrt(mu / r2) * (1 - np.sqrt((2 * r1) / (r1 + r2))))
        return v1 + v2

    def plane_change(i, v):
        return 2 * v * np.sin(i / 2)

    print(hohmann(200, 569))
    print(hohmann(414.1, 569) + plane_change(0.9014 - 0.4969, 7.59))

    r1 = 569 + 6371  # km
    r2 = 100 + 6371  # km
    mu = 398600.4
    v1 = abs(np.sqrt(mu / r1) * (np.sqrt((2 * r2) / (r1 + r2)) - 1))
    print(v1)


def problem2():
    mu = 398600.4
    R_E = 6371  # km

    def hst_parameters():
        # HST Parameters
        i = 0.4969  # rad
        Omega = 1.5199  # rad
        e = 0.0002935
        w = 2.8841  # rad
        h = 52519.6585  # km^2/s
        a = 6920  # km
        return i, Omega, e, w, h, a

    def iss_parameters():
        # ISS
        # 1 25544U 98067A   16039.53026564  .00016717  00000-0  10270-3 0  9007
        # 2 25544  51.6420 344.6243 0006534  89.0317 271.1585 15.54512915 24814

        i = np.deg2rad(51.6420)  # rad
        Omega = np.deg2rad(344.6243)  # rad
        e = 0.0006534
        w = np.deg2rad(89.0317)  # rad
        a = 6782  # km
        h = (a * (1 - e**2) * mu)**(1 / 2)   # km^2/s
        return i, Omega, e, w, h, a

    def gps_parameters():
        # Navstar 43
        i = np.deg2rad(55.7)  # rad
        Omega = np.deg2rad(246.5)  # rad
        e = 0.004456
        w = np.deg2rad(115)  # rad
        a = 26560  # km
        h = (a * (1 - e**2) * mu)**(1 / 2)   # km^2/s
        return i, Omega, e, w, h, a

    def eclipsed(D=0):
        D_0 = 79.0  # day number
        L_sun = np.deg2rad(((D - D_0) / 365) * 365)  # rad

        res = []
        for theta in np.deg2rad(np.linspace(0, 360, 3610)):
            r_sun = np.array([np.cos(L_sun),
                              np.sin(L_sun) * np.cos(np.deg2rad(23.45)),
                              np.sin(L_sun) * np.sin(np.deg2rad(23.45))])
            r_sat = np.array([np.cos(Omega) * np.cos(theta + w) - np.sin(Omega) * np.cos(i) * np.sin(theta + w),
                              np.sin(Omega) * np.cos(theta + w) +
                              np.cos(Omega) * np.cos(i) * np.sin(theta + w),
                              np.sin(i) * np.sin(theta + w)])

            r = (h**2 / mu) / (1 + e * np.cos(theta))
            beta = np.arcsin(R_E / r)
            eclipsed = (np.arccos(np.dot(-r_sun, r_sat)) <= beta)
            res.append([theta, *r_sun, *r_sat, r, beta, eclipsed])
        res = pd.DataFrame(res)
        res.columns = ['theta', 'sun_x', 'sun_y', 'sun_z',
                       'sat_x', 'sat_y', 'sat_z', 'r', 'beta', 'eclipsed']
        return pd.DataFrame(res)

    def check_days():
        res = []
        for d in np.linspace(0, 365, 366):
            df = eclipsed(d)
            df['Day'] = d
            res.append(df)
        return pd.concat(res).reset_index(drop=True)

    def theta_to_time(theta):
        E = 2 * np.arctan(np.tan(theta / 2) / np.sqrt((1 + e) / (1 - e)))
        M = E - e * np.sin(E)
        t = M / np.sqrt(mu / a**3)
        return t

    def mean_eclipsed_time():
        df = check_days()
        df['value_grp'] = (df.eclipsed.diff(1) != 0).astype('int').cumsum()
        crossings = pd.DataFrame({'Begin': df.groupby('value_grp').theta.first(),
                                  'End': df.groupby('value_grp').theta.last()}).reset_index(drop=True)
        crossings['Diff'] = abs(crossings['Begin'] - crossings['End'])
        crossings['t'] = crossings.Diff.apply(theta_to_time)
        return crossings[1:-2]

    i, Omega, e, w, h, a = hst_parameters()
    hst = mean_eclipsed_time()
    print(hst.t.abs().mean() / 60, hst.t.abs().sem() / 60,
          hst.t.abs().sum() / (60 * 60), len(hst))

    i, Omega, e, w, h, a = iss_parameters()
    iss = mean_eclipsed_time()
    print(iss.t.abs().mean() / 60, iss.t.abs().sem() / 60,
          iss.t.abs().sum() / (60 * 60), len(iss))

    i, Omega, e, w, h, a = gps_parameters()
    gps = mean_eclipsed_time()
    print(gps.t.abs().mean() / 60, gps.t.abs().sem() / 60,
          gps.t.abs().sum() / (60 * 60), len(gps))


def problem3():
    res = []
    for e in np.linspace(0, .002, 10000):
        mu = 398600.4
        C_D = 2.2
        rho = (4.1E-14 + 2E-13 + 8.5E-13 + 1.53E-12)/4  # kg/m^3 (solar max, night)

        a = 6920  # km
        A = 1E-4  # km^2
        M = 1000  # kg
        H = (60.4 + 67.4 + 74.6 + 88.7)/4  # km (solar max, night)

        def B():
            return ((mu / (a**3)) ** (0.5)) * ((A * C_D) / M) * (rho * a * e) * j1((a * e) / H) * np.exp(-e * (1 + (a / H)))

        def tau():
            return ((e**2) / (2 * B())) * (1 - (11 / 6) * e + (29 / 16) * e**2 + (7 / 8) * (H / a))

        days = tau() / (60 * 60 * 24)
        res.append([e, days])
    pd.DataFrame(res).plot(x=0, y=1)
    plt.show()


def problem4a():
    # Values for Navstar 43
    mu = 398600.4
    t0 = 0
    t = np.linspace(0, 60*60*24*7, 10000)

    i = np.deg2rad(55.7)  # rad
    Omega = np.deg2rad(246.5)  # rad
    e = 0.004456
    w = np.deg2rad(115)  # rad
    A = 26560  # km
    da = 60


    r = (A + da -
         A * e * np.cos((t-t0) * (mu/A**3)**(0.5)))
    lon = (Omega + w -
           t0 * (mu/A**3)**(0.5) -
           3/2 * da/A * (t - t0) * (mu/A**3)**(0.5) +
           2 * e * np.sin((t-t0) * (mu/A**3)**(0.5)))
    lat = i * np.sin(w + (t-t0) * (mu/A**3)**(0.5))
    df = pd.DataFrame([t, r, lon, lat]).T
    df.plot(x=2, y=3)
    plt.axis('equal')
    plt.show()

def problem4c():
    # Time in seconds for longitudinal station keeping at long = -100 deg

    f = 8.1E-9  # m/s^2
    r = 42E6  # m
    err_box = np.deg2rad(0.22)
    dV = 4 * ((r * f * err_box)/3) ** (0.5)
    T = 4 * ((r * err_box) / (3 * f)) ** (0.5)

    total_dV = 200
    return total_dV/(dV/T)

def problem5():
    r1, v1 = sv_from_coe(np.array([52059, 0.0257240, np.deg2rad(40), np.deg2rad(60), np.deg2rad(30), np.deg2rad(40)]), mu)
    r2, v2 = sv_from_coe(np.array([52362, 0.0072696, np.deg2rad(40), np.deg2rad(50), np.deg2rad(120), np.deg2rad(40)]), mu)
    r = r2 - r1
    omega = np.deg2rad(np.rad2deg(np.cross(r1, v1) / (np.linalg.norm(r1))**2))
    v = v2-v1 - np.cross(omega, r)
    print(r, v)


def problem6():
    def time_history(tf=5557.2, x=0, z=0, xdot=0, zdot=0, y=0, ydot=0, small=False):
        state_old = np.array([x, z, xdot, zdot, y, ydot])

        times = np.linspace(0, tf, 10000)
        n = 0.0011308  # rad/s ISS

        res = []
        if not small:
            for t in times:
                nt = n * t
                state = np.array([
                         [1,  6*(sin(nt) - nt), 4*sin(nt)/n - 3*t, 2*(cos(nt) - 1)/n, 0, 0],
                         [0,     4 - 3*cos(nt),   2*(1-cos(nt))/n,         sin(nt)/n, 0, 0],
                         [0, 6*n*(cos(nt) - 1),     4*cos(nt) - 3,      -2 * sin(nt), 0, 0],
                         [0,       3*n*sin(nt),         2*sin(nt),           cos(nt), 0, 0],
                         [0,                 0,                0, 0,    cos(nt), sin(nt)/n],
                         [0,                 0,                0, 0, -n*sin(nt),   cos(nt)],
                         ]) @ state_old
                state_old = state
                res.append([t, *state])
        else:
            raise NotImplemented

        res = pd.DataFrame(res)
        res.columns = ['t', 'x', 'z', 'xdot', 'zdot', 'y', 'ydot']
        return res

