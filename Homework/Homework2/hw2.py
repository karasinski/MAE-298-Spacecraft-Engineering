import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate.odepack import odeint
from scipy.integrate import ode


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
