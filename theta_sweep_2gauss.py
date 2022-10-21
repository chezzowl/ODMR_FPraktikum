"""
Handles measurements during the magnetic field sweep of varying theta at a constant phi.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.legend_handler
import numpy as np
from scipy.optimize import curve_fit
from functions import gaussian, two_gaussians, thPrime, dthPrime_dd, dthPrime_de, dthPrime_dnu
import constants as C

# ######################## VARIABLES TO PRESET MANUALLY ####################### #
# measurement number range
msr_start = 49
msr_end = 67

# naming conventions
measDir = "measurements\\"
prefix = "Measurement_"
ending = ".dat"

# flag deciding whether to save fit data at the end of the script
savemode = 0
# ############################################################################# #

# list containing all plots we want to scroll through
allplots_thetas = []
# 4-gaussian fit for each slice in heat map
allfits = []
# all_fit_params - list of tuples containing the pairs (fitParams, fitCovar) for each plot
allparams = []
# sorted array containing the locations of the four peaks encountered throughout the heat map
allpeaks = []
# sorted errors of peaks from gaussian fits
int_errors = []
# list containing all phi values
thetas = []

# flag, deciding whether to show all gaussians in all plots
show_all_gauss = 1

# ---------------- manually set guesses for the fit function in each slice of the heat map ---------------- #
sig12 = 0.01  # est. HWHM of first peak in first slice (which is the sum of peaks 1 and 2)
sig34 = 0.01  # est. HWHM of second peak in first slice (which is the sum of peaks 3 and 4)
A12 = (-0.016 / 2) * np.sqrt(2 * np.pi * sig12)
# est. half amplitude of first peak in first slice
A34 = (-0.017 / 2) * np.sqrt(2 * np.pi * sig34)
# est. half amplitude of second peak in first slice
offset = 0.9994
# small deltas for manual parameter tweaking
dA = 0.001
dh = 0.002
dsig = 0.0005

# format of the guess: [A1,x01,sig1,A2,x02,sig2,A3,x03,sig3,A4,x04,sig4,offset]
guesses = [
    [A12, 2.840, sig12, A34, 2.898, sig34, offset],  # 90
    [A12, 2.840, sig12, A34, 2.898, sig34, offset],  # 95
    [A12, 2.839, sig12, A34, 2.898, sig34, offset],  # 100
    [A12, 2.838, sig12, A34, 2.898, sig34, offset],  # 105
    [A12, 2.839, sig12, A34, 2.899, sig34, offset],  # 110
    [A12, 2.839, sig12, A34, 2.898, sig34, offset],  # 115
    [A12, 2.841, sig12, A34, 2.896, sig34, offset],  # 120
    [A12, 2.841, sig12, A34, 2.895, sig34, offset],  # 125
    [A12, 2.842, sig12, A34, 2.894, sig34, offset],  # 130
    [A12, 2.845, sig12, A34, 2.893, sig34, offset],  # 135
    [A12, 2.846, sig12, A34, 2.892, sig34, offset],  # 140
    [A12, 2.847, sig12, A34, 2.889, sig34, offset],  # 145
    [A12, 2.850, sig12, A34, 2.888, sig34, offset],  # 150
    [A12, 2.851, sig12, A34, 2.887, sig34, offset],  # 155
    [A12, 2.853, sig12, A34, 2.884, sig34, offset],  # 160
    [A12, 2.856, sig12, A34, 2.882, sig34, offset],  # 165
    [A12, 2.858, sig12, A34, 2.880, sig34, offset],  # 170
    [A12, 2.860, sig12, A34, 2.879, sig34, offset],  # 175
    [A12, 2.861, sig12, A34, 2.877, sig34, offset]  # 180
]
# -------------------------------- read files corresponding to the theta sweep -------------------------------- #
for i in range(msr_start, msr_end + 1):
    # generate file names specific to our measurement
    msr_str = str(i)
    datname = measDir + prefix + msr_str + ending
    print(datname)

    # read in file
    df = pd.read_csv(datname, skiprows=4, sep='\t', header=None)
    freq = df.iloc[:, 0].to_numpy() * 1e-9
    intens = df.iloc[:, 4].to_numpy()
    allplots_thetas.append((freq, intens))
    # save theta information, read from the second line in the file
    theta = round(pd.read_csv(datname, skiprows=1, sep='\t', header=None, nrows=1)[1].to_numpy()[0])
    thetas.append(theta)

    # ----------------------- fitting ----------------------- #
    guess = guesses[i - msr_start]
    fitParams, fitCovar = curve_fit(two_gaussians, freq, intens,
                                    p0=guess, maxfev=100000)
    # save all necessary arrays for later
    allfits.append((freq, two_gaussians(freq, *fitParams)))
    # append [x0_1, x0_2] to peak-center array
    allpeaks.append([fitParams[1], fitParams[4]])
    # append [sigma1, sigma2] to error array
    int_errors.append([fitParams[2], fitParams[5]])
    # append param information, used if gauss mode is ON
    allparams.append(fitParams)

# set initial position in left plot
curr_pos = 0

# convert everything to numpy arrays
allplots_thetas = np.asarray(allplots_thetas)
allfits = np.asarray(allfits)
allpeaks = np.asarray(allpeaks)
int_errors = np.asarray(int_errors)
thetas = np.asarray(thetas)
allparams = np.asarray(allparams)

# peak-specific colors
c1 = 'tab:orange'
c2 = 'tab:blue'
c3 = 'tab:green'
c4 = 'tab:red'


def key_event(e):
    """
    handler for key events on scrollable plot
    :param e: event being registered
    """
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(allplots_thetas)

    ax.cla()
    # ax.plot(allplots_thetas[curr_pos][0], allplots_thetas[curr_pos][1])
    ax.scatter(allplots_thetas[curr_pos][0], allplots_thetas[curr_pos][1], marker="x")
    ax.plot(allfits[curr_pos][0], allfits[curr_pos][1], color='magenta')
    # if selected, plot individual gaussians
    if show_all_gauss:
        # if gauss mode is on, plot all individual gaussians resulting from the fit function
        g1 = gaussian(allfits[curr_pos][0], allparams[curr_pos][0], allparams[curr_pos][1],
                      allparams[curr_pos][2]) + offset
        g2 = gaussian(allfits[curr_pos][0], allparams[curr_pos][3], allparams[curr_pos][4],
                      allparams[curr_pos][5]) + offset
        ax.plot(allfits[curr_pos][0], g1, '--', color=c1)
        ax.plot(allfits[curr_pos][0], g2, '--', color=c3)
    ax.set_title("$\phi = 0°; \\theta$ = " + str(thetas[curr_pos]) + "°")
    ax.set(ylim=[0.981, 1.003])
    ax.set_ylabel("Normalized Intensity")
    ax.set_xlabel("RF [GHz]")
    fig.canvas.draw()


# making everything prettier
plt.rc('font', size=7)  # controls default text sizes
plt.rc('axes', titlesize=20)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
plt.rc('figure', titlesize=4)  # fontsize of the figure title
plt.rcParams['axes.linewidth'] = 2
# ------------------------ figure 1: all slices with gaussians, scrollable ------------------------ #
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)
ax.scatter(allplots_thetas[curr_pos][0], allplots_thetas[curr_pos][1], marker="x")
ax.plot(allfits[curr_pos][0], allfits[curr_pos][1], color='magenta')
if show_all_gauss:
    # if gauss mode is on, plot all individual gaussians resulting from the fit function
    g1 = gaussian(allfits[curr_pos][0], allparams[curr_pos][0], allparams[curr_pos][1], allparams[curr_pos][2]) + offset
    g2 = gaussian(allfits[curr_pos][0], allparams[curr_pos][3], allparams[curr_pos][4], allparams[curr_pos][5]) + offset
    ax.plot(allfits[curr_pos][0], g1, '--', color=c1)
    ax.plot(allfits[curr_pos][0], g2, '--', color=c3)

ax.set_title("$\phi=0°; \\theta$ = " + str(thetas[curr_pos]) + "°")
ax.tick_params(width=2.5, length=10, top=True, right=True, direction='in')
ax.set(ylim=[0.981, 1.003])
ax.set_ylabel("Normalized Intensity")
ax.set_xlabel("RF [GHz]")

# ------------------------ figure 2 - heat map recreation from gaussians ------------------------ #
fig2 = plt.figure()
# plot peak positions with error bars
ax2 = fig2.add_subplot(111)

peak1_overtheta = allpeaks[:, 0]
peak2_overtheta = allpeaks[:, 1]
peak1_err = int_errors[:, 0]
peak2_err = int_errors[:, 1]

# plot peak position over theta
pk1 = ax2.errorbar(thetas, peak1_overtheta, yerr=peak1_err, ls='none', marker='x', capsize=3, ms=10,
                   label=r"for the [1,1,1] and [$\bar{1},1,\bar{1}$] NV axes")
pk2 = ax2.errorbar(thetas, peak2_overtheta, yerr=peak2_err, ls='none', marker='x', capsize=3, ms=10)
# ---------------- some extra tweaking for the legend --------------------- #
# Create empty plot with blank marker containing extra label we need in the beginning
dummyplot, = ax2.plot([], [], ' ', label="Gaussian fitted intensity minima")
dummyplot_2, = ax2.plot([], [], ' ')
handles = [(dummyplot, dummyplot_2), (pk1[0], pk2[0])]
_, labels = ax2.get_legend_handles_labels()
# use them in the legend
ax2.legend(handles=handles, labels=labels, loc='best', fontsize=15, markerscale=1.25,
           handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None)})

# --------- make figures pretty
ax2.set_ylabel("RF[GHz]")
ax2.set_xlabel("$\\theta$ [°]")
ax2.tick_params(width=2.5, length=10, top=True, right=True, direction='in')

plt.show()

# ------------ save peak information to be used in simulation ------------------ #
if savemode:
    data_to_save = np.asarray([thetas, peak1_overtheta, peak1_err, peak2_overtheta, peak2_err])
    # creta or overwrite file and save peaks info
    with open('data/peaks_from_theta.npy', 'wb') as f:
        np.save(f, data_to_save)

# ------- get B-projection angle theta' as a function of theta ------- #
# restore units
peak1_overtheta = peak1_overtheta * 10 ** 9
peak2_overtheta = peak2_overtheta * 10 ** 9
peak1_err = peak1_err * 10 ** 9
peak2_err = peak2_err * 10 ** 9

# 1) [111] - axis
nv1_thetaPrime_best = thPrime(peak1_overtheta + np.abs(peak1_err))  # best value array
nv1_thetaPrime_error = np.sqrt(
    dthPrime_de(peak1_overtheta) ** 2 * C.dE ** 2 + dthPrime_dd(peak1_overtheta) ** 2 * C.dD ** 2
    + dthPrime_dnu(peak1_overtheta) ** 2 * peak1_err ** 2)
# 2) [-1 1 -1] - axis
nv2_thetaPrime_best = thPrime(peak2_overtheta - np.abs(peak2_err))  # best value array
nv2_thetaPrime_error = np.sqrt(
    dthPrime_de(peak2_overtheta) ** 2 * C.dE ** 2 + dthPrime_dd(peak2_overtheta) ** 2 * C.dD ** 2
    + dthPrime_dnu(peak2_overtheta) ** 2 * peak2_err ** 2)

# TESTING
fig2 = plt.figure()
axtest = fig2.add_subplot(111)
axtest.errorbar(thetas, nv1_thetaPrime_best, yerr=nv1_thetaPrime_error, ls='none', marker='x', capsize=3)
