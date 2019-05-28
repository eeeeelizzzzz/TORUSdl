import numpy as np
from numpy import sin, cos

from __init__ import FILL_VALUE


def calc_homogeneity(raw_vr, derived_vr):
    """
    Determines homogeneity of the wind field as described in E. Paschke et. al. 2015 section 2.2.4
    :param raw_vr: Raw radial velocity
    :param derived_vr: Radial velocity derived from wind retrieval
    :return:
    """

    vr_bar = np.sum(raw_vr)

    return 1 - np.sum((raw_vr - derived_vr)**2) / np.sum((raw_vr - vr_bar)**2)


def calc_vad(az, elev, vel):
    """
    Performs a 2d (horizontal) VAD retrieval
    :param az:
    :param elev:
    :param vel:
    :return:
    """
    elev = np.deg2rad(elev)
    az = np.deg2rad(az)

    if vel.size > 1:  # If there could be sufficient data points...
        A = sum(vel * sin(az))
        B = sum(sin(az) ** 2) * cos(elev)
        C = sum(cos(az) * sin(az)) * cos(elev)

        D = sum(vel * cos(az))
        E = sum(sin(az) * cos(az)) * cos(elev)
        F = sum(cos(az) ** 2) * cos(elev)

        # solve A = uB + vC and D = uE + vF
        y = np.array([[B, E], [C, F]])
        z = np.array([A, D])
        # print y
        # print z
        try:
            sol = np.linalg.solve(y, z)
            # print sol
            u = sol[0]
            v = sol[1]
            return u, v
        except np.linalg.linalg.LinAlgError:
            return FILL_VALUE, FILL_VALUE, FILL_VALUE
    else:
        return FILL_VALUE, FILL_VALUE, FILL_VALUE


def calc_vad_3d(az, elev, vel):
    """
    Calculates the 3D VAD
    :param az: Azimuth data
    :param elev: Elevation Data
    :param vel: Velocity Data
    :return:
        u: U component of the wind
        v: V component of the wind
        w: W component of the wind
    """
    elev = np.deg2rad(elev)
    az = np.deg2rad(az)

    if vel.size > 1:  # If there could be sufficient data points...
        A = sum(vel * sin(az))
        B = sum(sin(az) ** 2 * cos(elev))
        C = sum(cos(az) * sin(az) * cos(elev))
        G = sum(sin(az) * sin(elev))

        D = sum(vel * cos(az))
        E = sum(sin(az) * cos(az) * cos(elev))
        F = sum(cos(az) ** 2 * cos(elev))
        H = sum(cos(az) * sin(elev))

        W = sum(vel)
        X = sum(sin(az)*cos(elev))
        Y = sum(cos(az)*cos(elev))
        Z = sum(az*sin(elev))

        # solve A = uB + vC + wG , D = uE + vF + wH and W = uX + vY+ wZ
        y = np.array([[B, E, X], [C, F, Y], [G, H, Z]])
        z = np.array([A, D, W])
        # print y
        # print z
        try:
            sol = np.linalg.solve(y, z)
            # print sol
            u = sol[0]
            v = sol[1]
            w = sol[2]
            return u, v, w
        except np.linalg.linalg.LinAlgError:
            return FILL_VALUE, FILL_VALUE, FILL_VALUE
    else:
        return FILL_VALUE, FILL_VALUE, FILL_VALUE