import numpy as np
from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt


# Origin is the tip of far center of the far end of Section1
Part = namedtuple('Part', ['mass', 'v1', 'v2', 'v3'])
Cylinder = namedtuple('Cylinder', Part._fields + ('length', 'radius'))
Plate = namedtuple('Plate', Part._fields + ('length', 'width', 'theta'))

Section1 = Cylinder(mass=9033, v1=309.25/2, v2=0, v3=0, length=309.25, radius=121.2/2)
Section2 = Cylinder(mass=10593, v1=309.25 + 61.25/2, v2=0, v3=0, length=61.25, radius=121.2/2)
Section3 = Cylinder(mass=3363, v1=309.25 + 61.25 + 138/2, v2=0, v3=0, length=138, radius=168.16/2)
Solar1 = Plate(mass=735/2, v1=309.25 - 20.75, v2=129 + 113.5/2, v3=0, length=476.8, width=113.5, theta=0)
Solar2 = Plate(mass=735/2, v1=309.25 - 20.75, v2=-129 - 113.5/2, v3=0, length=476.8, width=113.5, theta=0)

parts = [Section1, Section2, Section3, Solar1, Solar2]
r_cm = ([part.mass * np.array([part.v1, part.v2, part.v3]) for part in parts] /
        np.sum([part.mass for part in parts])).sum(axis=0)
print(r_cm)

def parallel_axis(part, inertia, r=[0, 0, 0]):
    # The moment of inertia about the center of mass of the body with respect
    # to an orthogonal coordinate system.
    Ic = inertia(part)
    m = part.mass

    # The distances along the three ordinates that located the new point
    # relative to the center of mass of the body.
    d = np.array([part.v1, part.v2, part.v3])

    a = d[0] - r[0]
    b = d[1] - r[1]
    c = d[2] - r[2]
    dMat = np.zeros((3, 3), dtype=object)
    dMat[0] = np.array([b**2 + c**2, -a * b, -a * c])
    dMat[1] = np.array([-a * b, c**2 + a**2, -b * c])
    dMat[2] = np.array([-a * c, -b * c, a**2 + b**2])
    return Ic + m * dMat


def SolidCylinder(cylinder):
    m = cylinder.mass
    r = cylinder.radius
    h = cylinder.length
    I = np.array([[1/12 * m * (3 * r**2 + h**2), 0, 0],
                  [0, 1/12 * m * (3 * r**2 + h**2), 0],
                  [0, 0, 1/2 * m * r**2]])
    return I


def ThinWalledCylinder(cylinder):
    m = cylinder.mass
    r = cylinder.radius
    h = cylinder.length
    I = np.array([[1/12 * m * (3 * 2*(r**2) + h**2), 0, 0],
                  [0, 1/12 * m * (3 * 2*(r**2) + h**2), 0],
                  [0, 0, 1/2 * m * 2 * (r**2)]])
    return I


def FlatPlate(plate):
    m = plate.mass
    a = plate.length
    b = plate.width
    I = np.array([[1/12 * m * b**2, 0, 0],
                  [0, 1/12 * m * a**2, 0],
                  [0, 0, 1/2 * m * (a**2 + b**2)]])
    return I


def TotalI(r, theta=0):
    I_S1 = parallel_axis(Section1, ThinWalledCylinder, r=r)
    I_S2 = parallel_axis(Section2, ThinWalledCylinder, r=r)
    I_S3 = parallel_axis(Section3, SolidCylinder, r=r)
    I_SA1 = parallel_axis(Solar1, FlatPlate, r=r)
    I_SA2 = parallel_axis(Solar2, FlatPlate, r=r)

    # Rotate if necessary
    I_SA1 = np.dot(rotation_matrix(np.array([0, 1, 0]), theta), I_SA1)
    I_SA2 = np.dot(rotation_matrix(np.array([0, 1, 0]), theta), I_SA2)
    # Convert from lb * inches^2 to kg * m^2
    I = (I_S1 + I_S2 + I_S3 + I_SA1 + I_SA2) * 0.000292639653
    return I

truth = np.array([[36046,  -706,  1491],
                  [ -706, 86868,   449],
                  [ 1491,   449, 93848]])
print(truth)
print(TotalI(r_cm))


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta)
    b, c, d = -axis*np.sin(theta)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def plot1():
    plt.close('all')
    res = []
    for theta in np.linspace(0, np.pi, 100):
        res.append([theta, *TotalI(r=r_cm, theta=theta).diagonal()])
    pd.DataFrame(res).plot(x=0, y=[1, 2, 3])
    plt.legend(['$I_{xx}$', '$I_{yy}$', '$I_{zz}$'], loc='upper right')
    plt.ylabel('Inertia [kg·m$^2$]')
    plt.xlabel('Solar Array Anle [rad/s]')
    plt.savefig('figure1.pdf')


def angularMomentum(w):
    ''' Using the true value for the inertia matrix.'''
    I = truth
    H = np.dot(I, w)
    return H

def plot2():
    plt.close('all')
    res = []
    for w in np.linspace(0, 0.1, 100):
        w *= np.ones(3)
        res.append([w[0], *angularMomentum(w)])
    pd.DataFrame(res).plot(x=0, y=[1, 2, 3])
    plt.legend(['$H_x$', '$H_y$', '$H_z$'], loc='upper left')
    plt.ylabel('Momentum [kg·m$^2$/s]')
    plt.xlabel('$\omega$ [rad/s]')
    plt.savefig('figure2.pdf')


