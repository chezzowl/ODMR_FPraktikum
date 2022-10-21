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

D = 2.869  # 1st ZFS parameter
E = 0.003  # 2nd ZFS parameter
h = 6.62607004e-34  # Planck-Konst in J/s
mu_B = 9.2740100783e-24  # Bohr'sches Magneton


def fitfunc(x, u1, u2, u3, u4, s1, s2, s3, s4, A, off):
    return (A / (s1 * np.sqrt(2 * np.pi)) * np.exp(-(x - u1) ** 2 / (2 * s1 ** 2)) + off +
            A / (s2 * np.sqrt(2 * np.pi)) * np.exp(-(x - u2) ** 2 / (2 * s2 ** 2)) + off +
            A / (s3 * np.sqrt(2 * np.pi)) * np.exp(-(x - u3) ** 2 / (2 * s3 ** 2)) + off +
            A / (s4 * np.sqrt(2 * np.pi)) * np.exp(-(x - u4) ** 2 / (2 * s4 ** 2)) + off)


def fitfunc2(B, a):
    return D + np.sqrt(E ** 2 + (a * B) ** 2)


B = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]
RF_up = []
d_RF_up = []
RF_down = []
d_RF_down = []

for i in range(0, 1):
    # file = 'Measurement_' + str(i+68) + '.DAT'
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
    plt.title(file, pad=25)
    if i == 4:
        guess = [2.851, 2.866, 2.872, 2.887, 0.002, 0.002,
                 0.002, 0.002, -0.008, 0.9985]
    else:
        guess = [2.866 - 0.025 * B[i], 2.8657, 2.8723, 2.872 + 0.025 * B[i], 0.0018, 0.0018,
                 0.0018, 0.0018, -0.012, 0.9985]
    fitParams, fitCovar = curve_fit(fitfunc, frequency, intensity,
                                    p0=guess, maxfev=100000)

    fit_x = np.arange(min(frequency), max(frequency), 0.0001)
    plt.plot(fit_x, fitfunc(fit_x, fitParams[0], fitParams[1], fitParams[2],
                            fitParams[3], fitParams[4], fitParams[5], fitParams[6],
                            fitParams[7], fitParams[8], fitParams[9]),
             color='red', lw=2, ls='-', label="Fit")

    RF_up.append(fitParams[3])
    d_RF_up.append(fitParams[7])
    RF_down.append(fitParams[0])
    d_RF_down.append(fitParams[4])

plt.show()

# fig, ax = plt.subplots(figsize=(18,12))#, constrained_layout=True)
# plt.errorbar(B, RF_up, d_RF_up, fmt='.', color='darkblue', capsize=3, markersize=1,
#               elinewidth=1.5, capthick=2, ecolor='darkblue')
# plt.errorbar(B, RF_down, d_RF_down, fmt='.', color='darkblue', capsize=3, markersize=1,
#               elinewidth=1.5, capthick=2, ecolor='darkblue')
# plt.xlabel('B [mT]', labelpad=8)
# plt.ylabel('Microwave frequency [GHz]', labelpad=15)
# plt.xticks(y=-0.01)
# plt.yticks(x=-0.005)
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
# # plt.xlim(min(theta), max(theta))
# # plt.ylim(8, max(N)*1.3)
# plt.tick_params(width=2.5, length=10, top=True, right=True, direction='in')
# plt.title('Zeeman Splitting', pad=25)
#
# RF_test = [D+(D-x) for x in RF_down]
# RF_fit = RF_up + RF_test
# d_RF_fit = d_RF_up + d_RF_down
# B_fit = B + B
# fitParams2, fitCovar2 = curve_fit(fitfunc2, B_fit, RF_fit,
#                                     p0=[0.028], maxfev=100000, sigma = d_RF_fit)
#
# fit_x2 = np.arange(0,3,0.001)
# plt.plot(fit_x2, fitfunc2(fit_x2, fitParams2[0]),
#           color='red', lw=2, ls='-', label="Fit")
# plt.plot(fit_x2, 2*D-fitfunc2(fit_x2, fitParams2[0]),
#           color='red', lw=2, ls='-')
#
# # Fitkonstante a von GHz/mT in Hz/T umrechnen
# a = fitParams2[0] * 1e12
# g_prime = h*a/mu_B
# print("g' = (g * cos(alpha)) =", g_prime)
#
# print(np.mean(RF_up + RF_down))
