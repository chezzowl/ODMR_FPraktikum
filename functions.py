"""
Holds a bunch of functions used throughout the scripts
"""
import numpy as np
import constants as C


# -------------- fitting related -------------- #

def gaussian(x, A, x0, sig):
    """

    :param x: input variable
    :param A: amplitude
    :param x0:
    :param sig: standard deviation
    :return:
    """
    return A / (sig * np.sqrt(2 * np.pi)) * np.exp(-(x - x0) ** 2 / (2 * sig ** 2))


def two_gaussians(x, *pars):
    """

    :param x:
    :param pars: format for parameters - [A1,x01,sig1,A2,x02,sig2,offset]
    :return:
    """
    if len(pars) is not 7:
        print(f"Warning: no offset specified in four_gaussian fit (Number of input parameters:{len(pars)}). "
              f"Offset is set to last input parameter!")
    off = pars[-1]
    ga1 = gaussian(x, pars[0], pars[1], pars[2])
    ga2 = gaussian(x, pars[3], pars[4], pars[5])
    return ga1 + ga2 + off


def three_gaussians(x, *pars):
    """

    :param x:
    :param pars: format for parameters - [A1,x01,sig1,A2,x02,sig2,A3,x03,sig3,offset]
    :return:
    """
    if len(pars) is not 10:
        print(f"Warning: no offset specified in four_gaussian fit (Number of input parameters:{len(pars)}). "
              f"Offset is set to last input parameter!")
    offset = pars[-1]
    g1 = gaussian(x, pars[0], pars[1], pars[2])
    g2 = gaussian(x, pars[3], pars[4], pars[5])
    g3 = gaussian(x, pars[6], pars[7], pars[8])
    return g1 + g2 + g3 + offset


def four_gaussians(x, *pars):
    """

    :param x:
    :param pars: format for parameters - [A1,x01,sig1,A2,x02,sig2,A3,x03,sig3,A4,x04,sig4,offset]
    :return:
    """
    if len(pars) is not 13:
        print(f"Warning: no offset specified in four_gaussian fit (Number of input parameters:{len(pars)}). "
              f"Offset is set to last input parameter!")
    offset = pars[-1]
    g1 = gaussian(x, pars[0], pars[1], pars[2])
    g2 = gaussian(x, pars[3], pars[4], pars[5])
    g3 = gaussian(x, pars[6], pars[7], pars[8])
    g4 = gaussian(x, pars[9], pars[10], pars[11])
    return g1 + g2 + g3 + g4 + offset


# ------------ error calculations -------------- #
def thPrime(freqs):
    """
    :param freqs: input frequency, single number or array-like
    :return: theta' (the B field - to - NV projection angle) in terms of input frequency
    """
    return np.arccos(np.sqrt(((freqs - C.D) ** 2 - C.E ** 2) / (C.alpha_factor ** 2 * C.B0 ** 2)))


def dthPrime_dnu(freqs):
    """
    :param freqs: input frequency, single number or array-like
    :return: theta' (the B field - to - NV projection angle) in terms of input frequency
    """
    return (C.D - freqs) / np.sqrt((C.D - C.E - freqs) * (C.D + C.E - freqs) * (
            C.alpha_factor ** 2 * C.B0 ** 2 - C.D ** 2 + 2 * C.D * freqs + C.E ** 2 - freqs ** 2))


def dthPrime_dd(freqs):
    """
    :param freqs: input frequency, single number or array-like
    :return: theta' (the B field - to - NV projection angle) in terms of input frequency
    """
    return (freqs - C.D) / np.sqrt((C.D - C.E - freqs) * (C.D + C.E - freqs) * (
            C.alpha_factor ** 2 * C.B0 ** 2 - C.D ** 2 + 2 * C.D * freqs + C.E ** 2 - freqs ** 2))


def dthPrime_de(freqs):
    """
    :param freqs: input frequency, single number or array-like
    :return: theta' (the B field - to - NV projection angle) in terms of input frequency
    """
    return C.E / np.sqrt((C.D - C.E - freqs) * (C.D + C.E - freqs) * (
            C.alpha_factor ** 2 * C.B0 ** 2 - C.D ** 2 + 2 * C.D * freqs + C.E ** 2 - freqs ** 2))

