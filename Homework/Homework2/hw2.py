import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate.odepack import odeint
from tabulate import tabulate


###############################################################################
def problem4():
    # Two impact types
    micro_meteroid = {'density': 0.5,
                      'velocity': 23,
                      'diameter': 0.1}

    orbital_debris = {'density': 7.8,
                      'velocity': 7,
                      'diameter': 0.1}

    # Two target materials
    alum_6061_t6 = {'BHN': 95,
                    'C_t': 5.433,
                    'density': 2.7}
    alum_7075_t6 = {'BHN': 150,
                    'C_t': 6.350,
                    'density': 2.81}

    def latex_output(func, theta=0):
        theta = np.deg2rad(theta)

        try:
            micro_6061 = "{:.3f}".format(func(micro_meteroid, alum_6061_t6, theta=theta))
            micro_7075 = "{:.3f}".format(func(micro_meteroid, alum_7075_t6, theta=theta))
            orbital_6061 = "{:.3f}".format(func(orbital_debris, alum_6061_t6, theta=theta))
            orbital_7075 = "{:.3f}".format(func(orbital_debris, alum_7075_t6, theta=theta))

            headers = ["Micro-meteroid", "Orbital Debris"]
            table = [['6061', micro_6061, orbital_6061],
                     ['7075', micro_7075, orbital_7075]]
        except TypeError:
            res = []
            for material in [micro_meteroid, orbital_debris]:
                for velocity in [1, 5, 10, 25, 100]:
                    temp = material.copy()
                    temp['velocity'] = velocity
                    r = "{:.3f}".format(func(temp, alum_6061_t6, bumper, theta=theta))
                    res.append(r)

            headers = ["Micro-meteroid", "Orbital Debris"]
            table = [['1km/s',   res[0], res[5]],
                     ['5km/s',   res[1], res[6]],
                     ['10km/s',  res[2], res[7]],
                     ['25km/s',  res[3], res[8]],
                     ['100km/s', res[4], res[9]]]

        print(tabulate(table, headers, tablefmt='latex'))

    def P_inf(particle, target, theta=0):
        theta = np.deg2rad(theta)
        if particle['density'] / target['density'] < 1.5:
            exp = 0.5
        else:
            exp = 2/3

        return 5.24 \
               * particle['diameter'] ** (19/18) \
               * target['BHN'] ** (-0.25) \
               * (particle['density']/target['density']) ** (exp) \
               * ((particle['velocity'] * np.cos(theta))/target['C_t']) ** (2/3)

    def wall_thickness(particle, target, theta=0):
        theta = np.deg2rad(theta)
        # for detached spall
        k = 2.2

        n = particle['diameter'] ** (19/18) \
            * target['BHN'] ** (0.25) \
            * (particle['density']/target['density']) ** (0.5)

        d = k * 5.24 \
            * ((particle['velocity'] * np.cos(theta))/target['C_t']) ** (2/3)

        return d/n

    def compare_t_and_p_inf(particle, target, theta=0):
        theta = np.deg2rad(theta)
        t = wall_thickness(particle, target, theta=0)
        p_inf = P_inf(particle, target, theta=0)
        comparison = t <= 2.2 * p_inf
        return comparison

    bumper = {'S': 10.2,
              'sigma': 57,
              'density': 2.84}

    def t_b(particle, target, theta=0):
        theta = np.deg2rad(theta)
        c_b = 0.2
        return c_b * particle['diameter'] \
               * (particle['density']/target['density'])

    def t_w(particle, target, bumper, theta=0):
        theta = np.deg2rad(theta)
        c_w = 0.16
        V_p = ((4/3) * np.pi * (particle['diameter']/2) ** 3)
        M_p = particle['density'] * V_p
        V_n = particle['velocity'] * np.cos(theta)
        return c_w * particle['diameter'] ** (0.5) \
               * (particle['density'] * target['density']) ** (1/6) \
               * (M_p) ** (1/3) \
               * (V_n/(bumper['S'] ** (0.5))) \
               * (70/bumper['sigma'])**0.5

    def d_c(particle, target, bumper, theta=0):
        theta = np.deg2rad(theta)
        tb = t_b(particle, target, theta)
        tw = t_w(particle, target, bumper, theta)

        if particle['velocity'] >= 7:
            dc = 3.918 * tw**(2/3) \
                  * particle['density']**(-1/3) \
                  * bumper['density']**(-1/9) \
                  * (particle['velocity'] * np.cos(theta))**(-2/3) \
                  * bumper['S']**(1/3) * (bumper['sigma']/70)**(1/3)
        elif particle['velocity'] <= 3:
            n = tw * (bumper['sigma']/40)**(0.5) + tb
            d = 0.6 * (np.cos(theta))**(5/3) \
                * particle['density']**(0.5) \
                * particle['velocity']**(2/3)
            dc = (n / d) ** (18/19)
        else:
            v_n = particle['velocity'] * np.cos(theta)
            dc_f = 1.071 * tw**(2/3) \
                   * particle['density']**(-1/3) \
                   * bumper['density']**(-1/9) \
                   * bumper['S']**(1/3) * (bumper['sigma']/70)**(1/3)

            n = tw * (bumper['sigma']/40)**(0.5) + tb
            d = 1.248 * particle['density']**(0.5) \
                * (np.cos(theta))
            dc_s = (n / d) ** (18/19)

            dc = dc_f * (v_n/4 - 0.75) + dc_s * (1.75 - v_n/4)

        return dc

###############################################################################
def prob5():
    def euler(f, y0, a, b, h):
        t, y = a, y0
        res = []
        while t <= b:
            res.append([t, y])
            t += h
            y += h * f(t, y)
        return res

    def rk4(f, y0, a, b, h):
        n = int((b - a)/h)
        vx = [0]*(n + 1)
        vy = [0]*(n + 1)
        vx[0] = x = a
        vy[0] = y = y0
        for i in range(1, n + 1):
            k1 = h*f(x, y)
            k2 = h*f(x + 0.5*h, y + 0.5*k1)
            k3 = h*f(x + 0.5*h, y + 0.5*k2)
            k4 = h*f(x + h, y + k3)
            vx[i] = x = a + i*h
            vy[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4)/6
        return vx, vy

    def func(t, y):
        return t + y

    xs = np.concatenate([np.arange(0, 1., 0.01), [1]])
    ys = [np.exp(x) - x - 1 for x in xs]
    exact = pd.DataFrame([xs, ys]).T
    exact.columns = ['t', 'y']
    hs = [0.01, 0.1, 0.5]
    styles = ['--', '-.', ':']

    f, ax = plt.subplots()
    exact.plot(x='t', y='y', ax=ax, style='k', label='Exact')
    for h, s in zip(hs, styles):
        res = euler(func, 0, 0, 1, h)
        res = pd.DataFrame(res)
        res.columns = ['t', 'y']
        res.plot(x='t', y='y', ax=ax, style='b' + s, label='Euler ' + str(h))
    plt.ylim(0, 0.8)
    plt.savefig('euler.pdf')

    f, ax = plt.subplots()
    exact.plot(x='t', y='y', ax=ax, style='k', label='Exact')
    for h, s in zip(hs, styles):
        vx, vy = rk4(func, 0, 0, 1, h)
        res = pd.DataFrame([vx, vy]).T
        res.columns = ['t', 'y']
        res.plot(x='t', y='y', ax=ax, style='r' + s, label='RK4 ' + str(h))
    plt.ylim(0, 0.8)
    plt.savefig('rk4.pdf')

    f, ax = plt.subplots()
    exact.plot(x='t', y='y', ax=ax, style='k', label='Exact')
    y0, t0, t1 = 0, 0, 1
    for h, s in zip(hs, styles):
        t = np.concatenate([np.arange(t0, t1, h), [t1]])
        y = odeint(func, [y0], t)[:, 0]
        res = pd.DataFrame([t, y]).T
        res.columns = ['t', 'y']
        res.plot(x='t', y='y', ax=ax, style='g' + s, label='odeint ' + str(h))
    plt.ylim(0, 0.8)
    plt.savefig('odeint.pdf')
