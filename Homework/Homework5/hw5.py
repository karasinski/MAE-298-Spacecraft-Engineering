import numpy as np
from numpy import sin, cos, pi
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
        return 2 * v * sin(i / 2)

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
            r_sun = np.array([cos(L_sun),
                              sin(L_sun) * cos(np.deg2rad(23.45)),
                              sin(L_sun) * sin(np.deg2rad(23.45))])
            r_sat = np.array([cos(Omega) * cos(theta + w) - sin(Omega) * cos(i) * sin(theta + w),
                              sin(Omega) * cos(theta + w) +
                              cos(Omega) * cos(i) * sin(theta + w),
                              sin(i) * sin(theta + w)])

            r = (h**2 / mu) / (1 + e * cos(theta))
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
        M = E - e * sin(E)
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
    e = 0.0002468
    mu = 3.986e14
    C_D = 2.2
    rho = (4.1E-14 + 2E-13 + 8.5E-13 + 1.53E-12)/4  # kg/m^3 (solar average)
    a = 6920.895e3 # m
    A = 4*4 # m^2
    M = 1000  # kg
    H = (60.4 + 67.4 + 74.6 + 88.7)/4 * 1E3 # km (solar average)

    B = ((mu / (a**3)) ** (0.5)) * ((A * C_D) / M) * (rho * a * e) * j1((a * e) / H) * np.exp(-e * (1 + (a / H)))

    tau = ((e**2) / (2 * B)) * (1 - (11 / 6) * e + (29 / 16) * e**2 + (7 / 8) * (H / a))
    print(tau/60/60/24)


def problem4a():
    # Values for Westar 1
    mu = 398600.4
    t = np.linspace(0, 60*60*24, 10000)

    i = np.deg2rad(14.15)  # rad
    Omega = np.deg2rad(336.8)  # rad
    e = 7.333E-4
    w = np.deg2rad(282)  # rad
    A = 42164.5  # km
    da = 42164.5 - 42269  # km

    mu_a = (mu/A**3)**(0.5)
    r = (A + da -
         A * e * cos(t * mu_a))
    lon = (Omega + w -
           3/2 * (da/A) * t * mu_a +
           2 * e * sin(t * mu_a))
    lat = i * sin(w + t * mu_a)
    df = pd.DataFrame([t, r, lon, lat]).T

    plt.close('all')
    f, ax = plt.subplots()
    plt.plot(df[2], df[3])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.xlim(lon[0], lon[-1])
    plt.savefig('p4a-1.pdf')
    #plt.show()

    plt.close('all')
    f, ax = plt.subplots(nrows=2, sharex=True)
    #ax[0].plot(df[0], df[1])
    #ax[0].set_ylabel('Geocentric Distance')
    ax[0].plot(df[0], df[2])
    ax[0].set_ylabel('Longitude')
    ax[1].plot(df[0], df[3])
    ax[1].set_ylabel('Latitude')
    ax[1].set_xlabel('Time (s)')
    plt.xlim(t[0], t[-1])
    plt.savefig('p4a-2.pdf')
    #plt.show()


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

        n = 0.0011308  # rad/s ISS
        t, dt = 0, 0.1
        nt = n * dt

        res = []
        if not small:
            res.append([t, *state_old])
            while t <= tf:
                state = np.array([
                         [1,  6*(sin(nt) - nt), 4*sin(nt)/n - 3*dt, 2*(cos(nt) - 1)/n, 0, 0],
                         [0,     4 - 3*cos(nt),    2*(1-cos(nt))/n,         sin(nt)/n, 0, 0],
                         [0, 6*n*(cos(nt) - 1),      4*cos(nt) - 3,      -2 * sin(nt), 0, 0],
                         [0,       3*n*sin(nt),          2*sin(nt),           cos(nt), 0, 0],
                         [0,                 0,               0, 0,      cos(nt), sin(nt)/n],
                         [0,                 0,               0, 0,   -n*sin(nt),   cos(nt)],
                         ]) @ state_old
                state_old = state
                res.append([t, *state])
                t += dt
        else:
            res.append([t, *state_old])
            while t <= tf:
                state = np.array([
                         [1,  6*((nt - (nt**3)/6) - nt), 4*(nt - (nt**3)/6)/n - 3*dt, 2*((1-(nt**2)/2) - 1)/n, 0, 0],
                         [0,     4 - 3*(1-(nt**2)/2),    2*(1-(1-(nt**2)/2))/n,         (nt - (nt**3)/6)/n, 0, 0],
                         [0, 6*n*((1-(nt**2)/2) - 1),      4*(1-(nt**2)/2) - 3,      -2 * (nt - (nt**3)/6), 0, 0],
                         [0,       3*n*(nt - (nt**3)/6),          2*(nt - (nt**3)/6),           (1-(nt**2)/2), 0, 0],
                         [0,                 0,               0, 0,      (1-(nt**2)/2), (nt - (nt**3)/6)/n],
                         [0,                 0,               0, 0,   -n*(nt - (nt**3)/6),   (1-(nt**2)/2)],
                         ]) @ state_old
                state_old = state
                res.append([t, *state])
                t += dt

        res = pd.DataFrame(res)
        res.columns = ['t', 'x', 'z', 'xdot', 'zdot', 'y', 'ydot']
        return res

    plt.close('all')
    f, ax = plt.subplots()
    exact = time_history(xdot=-.1)
    ax.plot(exact.x, exact.z, 'k--')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.savefig('6a.pdf')

    plt.close('all')
    f, ax = plt.subplots()
    exact = time_history(zdot=.1)
    ax.plot(exact.x, exact.z, 'k--')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.savefig('6b.pdf')

    plt.close('all')
    f, ax = plt.subplots()
    exact = time_history(xdot=-.1, zdot=.1)
    ax.plot(exact.x, exact.z, 'k--')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.savefig('6c.pdf')

    plt.close('all')
    f, ax = plt.subplots(ncols=2, figsize=(16,6))
    exact = time_history(xdot=-.1, small=False)
    approx = time_history(xdot=-.1, small=True)
    ax[0].plot(approx.x, approx.z, 'r')
    ax[0].plot(exact.x, exact.z, 'k--')
    ax[0].axis('equal')
    ax[0].set_xlabel('x (m)')
    ax[0].set_ylabel('z (m)')

    exact = time_history(zdot=.1, small=False)
    approx = time_history(zdot=.1, small=True)
    ax[1].plot(approx.x, approx.z, 'r')
    ax[1].plot(exact.x, exact.z, 'k--')
    ax[1].axis('equal')
    ax[1].set_xlabel('x (m)')
    ax[1].legend(['Approx', 'Exact'])
    plt.savefig('6dboth.pdf')


def problem7():
    mu = 398600.4
    R_E = 6371
    HST_altitude = 550
    a = R_E + HST_altitude
    drop_off_altitude = 200

    # Catch up altitude for four orbits
    #da = (a) * np.deg2rad(65/4)/(-3*pi)

    period_HST = 2*pi*((HST_altitude + R_E)**3/mu)**(0.5)  # seconds
    period_at_200km = 2*pi*((drop_off_altitude + R_E)**3/mu)**(0.5)
    period_mean = (period_HST + period_at_200km)/2

    # Phasing
    phase_reduction = -3 * pi * ((HST_altitude-drop_off_altitude)/(HST_altitude + R_E))
    distance = phase_reduction * (HST_altitude-drop_off_altitude)
    behind = np.deg2rad(65)
    n_orbits = abs(behind/phase_reduction)
    phasing_time = period_HST * int(n_orbits)
    phasing_dv = 0

    # Homing
    da = -10
    homing_burn = (-0.5 * (da/a) * (mu/a)**(0.5)) * 1E3
    homing_time = pi*(((a*(1+da/(2*a)))**3)/mu)**(0.5)

    # Closing
    k = 1
    cycloidal_burn = (2 * (.8/(6*pi*k)) * (mu/(a**3))**0.5) * 1E3

    phasing = time_history(tf=2*period_mean, z=-30E3)
    homing = time_history(tf=period_HST/2, x=1E3, xdot=-homing_burn/2)
    closing = time_history(tf=period_HST, x=1E3, xdot=-cycloidal_burn/2)

    f, ax = plt.subplots()
    ax.plot(phasing.x, phasing.z, label='Phasing', color='r')
    ax.plot(homing.x, homing.z, label='Homing', color='b')
    ax.plot(closing.x, closing.z, label='Closing', color='g')
    plt.legend(loc='best')
    #plt.gca().invert_xaxis()
    plt.show()
