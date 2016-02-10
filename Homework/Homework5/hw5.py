import numpy as np


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

    r1 = 569 + 6371
    r2 = 100 + 6371
    mu = 398600.4
    v1 = abs(np.sqrt(mu/r1) * (np.sqrt((2*r2)/(r1+r2)) - 1))
    print(v1)
