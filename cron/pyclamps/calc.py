import numpy as np

epsilon = .622


def vapor_pressure(pressure, mixing_ratio):
    return pressure * mixing_ratio / (epsilon + mixing_ratio)


def mixing_ratio(e, p):
    return .622 * e / (p - e)


def sat_mixing_ratio(p, t):
    return mixing_ratio(sat_vapor_pressure(t), p)


def dewpoint(e):
    """
    returns Td in C
    :param e:
    :return:
    """
    return 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))


def sat_vapor_pressure(T):
    """
    :param T: T in degrees C
    :return:
    """
    return 6.112 * np.exp(17.67 * T / (T + 243.5))


def potential_temp(temp, press):
    return temp * (1000. / press) ** .286


def wind_spd(u, v):
    return np.sqrt(u**2 + v**2)


def wind_dir(u, v):
    dir = np.arctan2(-u, -v)
    return np.where(dir < 0, np.rad2deg(dir) + 360, np.rad2deg(dir))