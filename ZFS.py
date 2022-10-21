import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.rc('font', size=12)  # controls default text sizes
plt.rc('axes', titlesize=42)  # fontsize of the axes title
plt.rc('axes', labelsize=42)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=34)  # fontsize of the tick labels
plt.rc('ytick', labelsize=34)  # fontsize of the tick labels
plt.rc('figure', titlesize=8)  # fontsize of the figure title
plt.rcParams['axes.linewidth'] = 2


def fitfunc(x, u1, u2, a1, a2, s1, s2, o):
    """
    :param x: input variable
    :param u1:
    :param u2:
    :param a1:
    :param a2:
    :param s1:
    :param s2:
    :param o:
    :return: sum of two gaussians with the given parameters
    """
    return (a1 / (s1 * np.sqrt(2 * np.pi)) * np.exp(-(x - u1) ** 2 / (2 * s1 ** 2)) +
            a2 / (s2 * np.sqrt(2 * np.pi)) * np.exp(-(x - u2) ** 2 / (2 * s2 ** 2)) + o)


file = 'measurements\\Measurement_3.DAT'
df = pd.read_csv(file, skiprows=3, sep='\t')

frequency = df.iloc[:, 0] * 1e-9  # RF in GHz
intensity = df.iloc[:, 4]  # rate intensity (counters)

fig, ax = plt.subplots(figsize=(18, 12))  # , constrained_layout=True)
plt.plot(frequency, intensity, 'b-')
plt.xlabel('Frequency [GHz]', labelpad=8)
plt.ylabel('Normalized ODMR-Intensity', labelpad=15)
plt.xticks(y=-0.01)
plt.yticks(x=-0.005)
plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='grey', linestyle='-', alpha=0.1)
# plt.xlim(450,950)
# plt.ylim(0.05, 0.35)
plt.tick_params(width=2.5, length=10, top=True, right=True, direction='in')
plt.title('Zero field splitting', pad=25)

guess = [2.8657, 2.8723, -9e-5, -9e-5, 0.0025, 0.0025, 0.997]
fitParams, fitCovar = curve_fit(fitfunc, frequency, intensity, p0=guess)

fit_x = np.arange(min(frequency), max(frequency), 0.0001)
plt.plot(fit_x, fitfunc(fit_x, fitParams[0], fitParams[1], fitParams[2],
                        fitParams[3], fitParams[4], fitParams[5],
                        fitParams[6]),
         color='red', lw=2, ls='-', label="Fit")

# Positions and standard deviations of Gaussians:
P1 = fitParams[0]
P2 = fitParams[1]
sigma1 = fitParams[4]
sigma2 = fitParams[5]

# zero field parameters in GHz
D = 0.5 * (P1 + P2)
delta_D = 0.5 * np.sqrt(sigma1 ** 2 + sigma2 ** 2)

E = abs(0.5 * (P1 - P2))
delta_E = 0.5 * np.sqrt(sigma1 ** 2 + sigma2 ** 2)

print('Zero field parameters:')
print('D =', D.round(4), '+-', delta_D.round(4), 'GHz')
E_MHz = E * 1000
d_E_MHz = delta_E * 1000
print('E =', E_MHz.round(2), '+-', d_E_MHz.round(2), 'MHz')

plt.show()
