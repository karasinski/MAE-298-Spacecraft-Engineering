import numpy as np
from numpy import cos, sin
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import solve


class Satellite(object):
    def __init__(self, state):
        # State is x, y, z, dx, dy, dz
        self.state = state

        mu = 398600.4418   # km^3/s^2; gravitational parameter of earth
        r_e = 6378.1       # km; radius of the earth
        height_hst = 550   # km; height of hubble
        n = np.sqrt(mu / (r_e + height_hst)**3)
        self.n = n
        self.t = 0

    def DeltaV(self, target_state, t):
        '''
        Find the delta_v required to take the system to zero for some time.
        Assume you start with a known position (x0, y0, z0).
        '''

        x0, y0, z0, dx0, dy0, dz0 = self.state
        xf, yf, zf, dxf, dyf, dzf = target_state

        n = self.n

        # solve Ax = b
        b = np.array([(3. * cos(n * t) - 4.) * (x0 - xf), (y0 - yf) + 6. * (x0 - xf) * (sin(n * t) - n * t)])

        A = np.array([[(1. / n) * sin(n * t), (2. / n) * (1. - cos(n * t))],
                      [(2. / n) * (1. - cos(n * t)), 3. * t - (4. * sin(n * t) / n)]])

        x = solve(A, b)

        xd0 = x[0]
        yd0 = x[1]
        zd0 = -n * (z0 - zf) * cos(n * t) / sin(n * t)

        return np.array([xd0, yd0, zd0])


    def ClohessyWiltshire(self, t):
        '''Clohessy-Wiltshire equations'''

        n = self.n
        x0, y0, z0, dx0, dy0, dz0 = self.state

        x = (4 - 3 * cos(n * t)) * x0 + (sin(n * t) / n) * dx0 + (2 / n) * (1 - cos(n * t)) * dy0
        y = 6 * (sin(n * t) - n * t) * x0 + y0 - (2 / n) * (1 - cos(n * t)) * dx0 + 1 / n * (4 * sin(n * t) - 3 * n * t) * dy0
        z = z0 * cos(n * t) + dz0 / n * sin(n * t)

        state = np.array([x, y, z, [dx0]*len(t), [dy0]*len(t), [dz0]*len(t)])
        return state

        #self.state = np.array([x, y, z, dx0, dy0, dz0])
        #self.t += t

#state = np.array([-800, 0, -2000, 0, 0, -3])
#t = np.linspace(0.0, 90*60, 100000)

state = np.array([-250E3, 0, 1E3, -1, 0, -1])
hrv = Satellite(state)

t = np.linspace(0.0, 3*60*60, 100000)
res = pd.DataFrame(np.array([t, *hrv.ClohessyWiltshire(t)])).T
res.columns = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz']
#res.plot(x='t', y=['x', 'y', 'z'])
#res.plot(x='x', y='z')
#plt.show()



f, ax = plt.subplots()
ax.xaxis.set_ticks_position('top') # the rest is the same

plt.plot(res.x, res.z)
plt.scatter(res.x[::10000], res.z[::10000], color='r')
#plt.title('ORBT RENDEZVOUS PROFILE')
#plt.xlim(0, -250)
#plt.ylim(0, 40)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.xlabel('Vbar')
plt.ylabel('Rbar')
ax.xaxis.set_label_position('top')

plt.show()


def make_me_a_plot(state, t):
    hrv = Satellite(state)

    t = np.linspace(0.0, t, 100000)
    res = pd.DataFrame(np.array([t, *hrv.ClohessyWiltshire(t)])).T
    res.columns = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz']

    f, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top') # the rest is the same

    plt.axhline()
    plt.axvline()

    plt.plot(res.x, res.z)
    plt.scatter(res.x[0], res.z[0], color='k')
    plt.scatter(res.x[::10000], res.z[::10000], facecolors='none', edgecolors='r')
    #plt.title('ORBT RENDEZVOUS PROFILE')
    #plt.xlim(0, -250)
    #plt.ylim(0, 40)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    plt.xlabel('Vbar')
    plt.ylabel('Rbar')
    ax.xaxis.set_label_position('top')
    plt.axis('equal')

    plt.show()
















def DeltaV(self, t):
    '''
    Find the delta_v required to take the system to zero for some time.
    Assume you start with a known position (x0, y0, z0).
    '''

    x0, y0, z0, dx0, dy0, dz0 = self.state
    xf, yf, zf, dxf, dyf, dzf = self.target_state

    n = self.n

    # solve Ax = b
    b = np.array([(3. * cos(n * t) - 4.) * (x0 - xf), (y0 - yf) + 6. * (x0 - xf) * (sin(n * t) - n * t)])

    A = np.array([[(1. / n) * sin(n * t), (2. / n) * (1. - cos(n * t))],
                  [(2. / n) * (1. - cos(n * t)), 3. * t - (4. * sin(n * t) / n)]])

    x = solve(A, b)

    xd0 = x[0]
    yd0 = x[1]
    zd0 = -n * (z0 - zf) * cos(n * t) / sin(n * t)

    self.optimal_velocity = np.array([xd0, yd0, zd0])

