import numpy as np
import pandas as pd


def problem1():
    def hohmann(r1, r2, mu=398600.4, re=6371):
        r1 += re
        r2 += re

        v1 = abs(np.sqrt(mu/r1) * (np.sqrt((2*r2)/(r1+r2)) - 1))
        v2 = abs(np.sqrt(mu/r2) * (1 - np.sqrt((2*r1)/(r1+r2))))
        return v1 + v2

    def plane_change(i, v):
        return 2 * v * np.sin(i/2)

    print(hohmann(200, 569))
    print(hohmann(414.1, 569) + plane_change(0.9014 - 0.4969, 7.59))

    r1 = 569 + 6371  # km
    r2 = 100 + 6371  # km
    mu = 398600.4
    v1 = abs(np.sqrt(mu/r1) * (np.sqrt((2*r2)/(r1+r2)) - 1))
    print(v1)


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
    h = (a*(1-e**2)*mu)**(1/2)   # km^2/s
    return i, Omega, e, w, h, a


def gps_parameters():
    # Navstar 43
    i = np.deg2rad(55.7)  # rad
    Omega = np.deg2rad(246.5)  # rad
    e = 0.004456
    w = np.deg2rad(115)  # rad
    a = 26560  # km
    h = (a*(1-e**2)*mu)**(1/2)   # km^2/s
    return i, Omega, e, w, h, a


def eclipsed(D=0):
    D_0 = 79.0  # day number
    L_sun = np.deg2rad(((D - D_0)/365) * 365)  # rad

    res = []
    for theta in np.deg2rad(np.linspace(0, 360, 3610)):
        r_sun = np.array([np.cos(L_sun),
                          np.sin(L_sun)*np.cos(np.deg2rad(23.45)),
                          np.sin(L_sun)*np.sin(np.deg2rad(23.45))])
        r_sat = np.array([np.cos(Omega)*np.cos(theta + w) - np.sin(Omega)*np.cos(i)*np.sin(theta + w),
                          np.sin(Omega)*np.cos(theta + w) + np.cos(Omega)*np.cos(i)*np.sin(theta + w),
                          np.sin(i)*np.sin(theta + w)])

        r = (h**2 / mu)/(1 + e*np.cos(theta))
        beta = np.arcsin(R_E/r)
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
    E = 2*np.arctan(np.tan(theta/2)/np.sqrt((1+e)/(1-e)))
    M = E - e*np.sin(E)
    t = M/np.sqrt(mu/a**3)
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
print(hst.t.abs().mean()/60, hst.t.abs().sem()/60, hst.t.abs().sum()/(60*60), len(hst))

i, Omega, e, w, h, a = iss_parameters()
iss = mean_eclipsed_time()
print(iss.t.abs().mean()/60, iss.t.abs().sem()/60, iss.t.abs().sum()/(60*60), len(iss))

i, Omega, e, w, h, a = gps_parameters()
gps = mean_eclipsed_time()
print(gps.t.abs().mean()/60, gps.t.abs().sem()/60, gps.t.abs().sum()/(60*60), len(gps))
