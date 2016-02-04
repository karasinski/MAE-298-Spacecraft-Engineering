import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('atmosphere.dat', delimiter=r"\s+")
f, ax = plt.subplots()
df.query('Year == 1998').plot(x="Height", y="Density", ax=ax, color='b', logy=True, label='1998 (Min)')
df.query('Year == 2002').plot(x="Height", y="Density", ax=ax, color='g', logy=True, label='2002 (Max)')
plt.axvline(400, ls='--', color='k')
plt.axvline(550, ls='--', color='k')
plt.xlabel('Height (km)')
plt.ylabel('Density, g/cm$^{-3}$')
plt.tight_layout()
plt.savefig('p5.pdf')

max_iss = df.query('Height == 400 and Year == 2002').Density.values[0]
max_hst = df.query('Height == 550 and Year == 2002').Density.values[0]
min_iss = df.query('Height == 400 and Year == 1998').Density.values[0]
min_hst = df.query('Height == 550 and Year == 1998').Density.values[0]

def D(rho, norm=True):
    rho /= 1000  # convert from g/cm-3 to kg/m-3
    S = 55.44  # m
    C = 2.5
    v = np.array([6.47, 1.97, -3.45]) * 1000  # convert from km/s to m/s
    d = -1/2 * rho * S * C * np.linalg.norm(v)**2 * (v/np.linalg.norm(v))

    if norm:
        return np.linalg.norm(d)
    else:
        return d


print('Max ISS ', D(max_iss))
print('Max HST ', D(max_hst))
print('Min ISS ', D(min_iss))
print('Min HST ', D(min_hst))

df['Drag'] = df.Density.apply(D)

f, ax = plt.subplots()
df.query('Year == 1998').plot(x="Height", y="Drag", ax=ax, color='b', logy=True, label='1998 (Min)')
df.query('Year == 2002').plot(x="Height", y="Drag", ax=ax, color='g', logy=True, label='2002 (Max)')
plt.axvline(400, ls='--', color='k')
plt.axvline(550, ls='--', color='k')
plt.xlabel('Height (km)')
plt.ylabel('Drag Force, N')
plt.tight_layout()
plt.savefig('p5b.pdf')
