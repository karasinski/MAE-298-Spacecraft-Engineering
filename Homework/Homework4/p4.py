import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def b():
    mu = 398600.4 #km^3/s^2
    J2 = 1.08263e-3
    i, w = 0.4969, 2.8841
    r = 6378 + 568
    theta = np.linspace(0, 2*np.pi, 1000)

    p_r = (-mu/(r**2)) * (3/2) * J2 * (1 - 3 * np.sin(i)**2 * np.sin(w + theta)**2)
    p_p = (-mu/(r**2)) * (3/2) * J2 * (np.sin(i)**2 * np.sin(2*(w + theta)))
    p_h = (-mu/(r**2)) * (3/2) * J2 * (np.sin(2*i) * np.sin(w + theta))

    res = [theta, p_r, p_p, p_h]

    plt.close('all')
    pd.DataFrame(res).T.plot(x=0, y=[1, 2, 3])
    plt.legend(['$r$', '$\perp$', '$h$'])
    plt.xlabel('$\\theta$')
    plt.ylabel('$p$')
    plt.tight_layout()
    plt.savefig('p4.pdf')
