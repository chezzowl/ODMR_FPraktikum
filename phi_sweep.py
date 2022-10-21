"""
Handles measurements during the magnetic field sweep of varying phi at a constant theta.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.legend_handler
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from functions import gaussian, two_gaussians, three_gaussians, four_gaussians, thPrime, dthPrime_dd, dthPrime_de, \
    dthPrime_dnu
import constants as C

# ######################## VARIABLES TO PRESET MANUALLY ####################### #
# measurement number range
msr_start = 30
msr_end = 48

# naming conventions
measDir = "measurements\\"
prefix = "Measurement_"
ending = ".dat"

# flag deciding whether to save fit data at the end of the script
savemode = 0
# ############################################################################# #

# list containing all plots we want to scroll through
allplots_phis = []
# 4-gaussian fit for each slice in heat map
allfits = []
# all_fit_params - list of tuples containing the pairs (fitParams, fitCovar) for each plot
allparams = []
# sorted array containing the locations of the four peaks encountered throughout the heat map
allpeaks = []
# sorted errors of peaks from gaussian fits
int_errors = []
# list containing all phi values
phis = []

# flag, deciding whether to show all gaussians in all plots
show_all_gauss = 1

# ---------------- manually set guesses for the fit function in each slice of the heat map ---------------- #
sig12 = 0.01 / 2  # est. HWHM of first peak in first slice (which is the sum of peaks 1 and 2)
sig34 = 0.01 / 2  # est. HWHM of second peak in first slice (which is the sum of peaks 3 and 4)
# est. half amplitude of first pair of peaks in first slice (which is the sum of peaks 1 and 2)
A12 = (-0.015 / 2) * np.sqrt(2 * np.pi * sig12)
# est. half amplitude of second first pair of peaks in first slice (which is the sum of peaks 3 and 4)
A34 = (-0.017 / 2) * np.sqrt(2 * np.pi * sig34)
offset = 0.9992
# small deltas for manual parameter tweaking
dA = 0.001
dh = 0.002
dsig = 0.0005

# format of the guess: [A1,x01,sig1,A2,x02,sig2,A3,x03,sig3,A4,x04,sig4,offset]
guesses = [
    # [A12, 2.84, sig12, A12, 2.84, sig12, A34, 2.90, sig34, A34, 2.90, sig34, offset],  # 0
    # [A12, 2.839 - 1.5 * dh, sig12, A12, 2.840, sig12, A34, 2.898 - dh, sig34, A34,
    #  2.900, sig34, offset],  # 5
    [A12, 2.84, sig12, A34, 2.90, sig34, offset],  # 0
    [A12, 2.839, sig12, A34, 2.900, sig34, offset],  # 5
    [A12, 2.836 - dh / 1.3, sig12, A12, 2.842 + dh / 1.4, sig12, A34, 2.896, sig34, A34, 2.90 + dh / 2, sig34, offset],
    # 10
    # [A12, 2.836-dh/1.5, sig12, A12, 2.842+dh/1.5, sig12, A34, 2.896, sig34, A34, 2.90 + dh/2, sig34, offset],  # 10 vertauscht
    [A12, 2.835 - dh, sig12, A12, 2.846 + dh, sig12, A34, 2.893, sig34, A34, 2.901, sig34, offset],  # 15
    [A12, 2.834, sig12, A12, 2.847, sig12, A34, 2.893, sig34, A34, 2.901, sig34, offset],  # 20
    [A12, 2.836, sig12, A12, 2.849, sig12, A34, 2.890, sig34, A34, 2.901, sig34, offset],  # 25
    [A12, 2.838, sig12, A12, 2.850, sig12, A34, 2.887, sig34, A34, 2.902, sig34, offset],  # 30
    [A12, 2.836, sig12, A12, 2.853, sig12, A34, 2.885, sig34, A34, 2.900, sig34, offset],  # 35
    [A12, 2.836, sig12, A12, 2.855, sig12, A34, 2.883, sig34, A34, 2.901, sig34, offset],  # 40
    [A12, 2.838, sig12, A12, 2.856, sig12, A34, 2.880, sig34, A34, 2.900, sig34, offset],  # 45
    [A12, 2.837, sig12, A12, 2.861, sig12, A34, 2.877, sig34, A34, 2.897, sig34, offset],  # 50
    [A12, 2.841, sig12, A12, 2.862, sig12, A34, 2.875, sig34, A34, 2.897, sig34, offset],  # 55
    # [A12, 2.841, sig12, A12, 2.865 - 1.5 * dh, sig12, A34, 2.872 + 1.5 * dh, sig34, A34, 2.896, sig34, offset],  # 60
    # [A12, 2.844, sig12, A12, 2.867 - dh, sig12, A34, 2.873 + dh, sig34, A34, 2.893, sig34, offset],  # 65
    [A12, 2.841, sig12, A12, 2.868, 2 * sig12, A34, 2.896, sig34, offset],  # 60
    [A12, 2.844, sig12, A12, 2.870, 2 * sig12, A34, 2.893, sig34, offset],  # 65
    [A12, 2.846, sig12, A12, 2.865 - dh, sig12, A34, 2.873 + dh, sig34, A34, 2.891, sig34, offset],  # 70
    [A12, 2.847, sig12, A12, 2.863, sig12, A34, 2.876, sig34, A34, 2.889, sig34, offset],  # 75
    [A12, 2.850 - dh, sig12, A12, 2.862 + dh, sig12, A34, 2.878 - dh, sig34, A34, 2.886 + dh, sig34, offset],  # 80
    # [A12, 2.854 - 1.3 * dh, sig12, A12, 2.856 + 1.3 * dh, sig12, A34, 2.880, sig34, A34, 2.884, sig34, offset],  # 85
    # [A12, 2.857 - dh, sig12, A12, 2.857 + dh, sig12, A34, 2.881 - dh, sig34, A34, 2.881 + dh, sig34, offset]  # 90
    [A12, 2.854, sig12, A34, 2.884, sig34, offset],  # 85
    [A12, 2.857, sig12, A34, 2.881, sig34, offset]  # 90
]

# -------------------------------- read files corresponding to the phi sweep -------------------------------- #
for i in range(msr_start, msr_end + 1):
    idx = i - msr_start
    # generate file names specific to our measurement
    msr_str = str(i)
    datname = measDir + prefix + msr_str + ending
    print(datname)

    # read in file
    df = pd.read_csv(datname, skiprows=4, sep='\t', header=None)
    freq = df.iloc[:, 0].to_numpy() * 1e-9  # RF in GHz
    intens = df.iloc[:, 4].to_numpy()
    # ------------------ gaussian blurring, JUST AS A TEST ------------------ #
    # properly set up gauss filter with window size 5
    sigma = 2
    windowsize = 5
    # properly set truncate variable
    trunc = t = (((windowsize - 1) / 2) - 0.5) / sigma
    intens_blurred = ndimage.filters.gaussian_filter(intens, truncate=trunc, sigma=sigma)
    # ----------------------------------------------------------------------- #
    # save all plots in (x,y) tuples and keep phi information
    allplots_phis.append((freq, intens))
    phi = round(pd.read_csv(datname, skiprows=1, sep='\t', header=None, nrows=1)[1].to_numpy()[0])
    phis.append(phi)

    # ----------------------- fitting ----------------------- #
    # !!! this works the way it is now JUST FOR OUR DATASET and has to be adjusted otherwise !!!
    guess = guesses[idx]
    # fitting depends on whether there are 2 or 4 gaussians in the current picture
    if idx is 0 or idx is 1 or idx is 17 or idx is 18:  # first two and last two measurements
        # ... the outer 2 peaks overlap here ===> 2 gaussians
        fitParams, fitCovar = curve_fit(two_gaussians, freq, intens,
                                        p0=guess, maxfev=100000)
        allfits.append((freq, two_gaussians(freq, *fitParams)))
        # here, we tweak the fitparams parameters in order to get them into the usual shape ...
        # i.e. [A1,x01,sig1,A2,x02,sig2,offset] --->[A1,x01,sig1,A1,x01,sig1,A2,x02,sig2,A2,x02,sig2,offset]
        fitParams = np.concatenate((fitParams[0:3], fitParams[0:3], fitParams[3:6], fitParams[3:6], [fitParams[-1]]))
    elif idx is 12 or idx is 13:  # at 60° and 65°
        # the outer 2 peaks are individual, the inner peak is the sum of 2
        fitParams, fitCovar = curve_fit(three_gaussians, freq, intens,
                                        p0=guess, maxfev=100000)
        allfits.append((freq, three_gaussians(freq, *fitParams)))
        # here, we tweak the fitparams parameters in order to get them into the usual shape ...
        # i.e. [A1,x01,sig1,A2,x02,sig2,A3,x03,sig3,offset] --->[A1,x01,sig1,A2,x02,sig2,A2,x02,sig2,A3,x03,sig3,offset]
        fitParams = np.concatenate((fitParams[0:3], fitParams[3:6], fitParams[3:6], fitParams[6:9], [fitParams[-1]]))
    else:  # we are not in the critical picture ==> fit 4 gaussians
        fitParams, fitCovar = curve_fit(four_gaussians, freq, intens,
                                        p0=guess, maxfev=100000)
        allfits.append((freq, four_gaussians(freq, *fitParams)))

    # append [x0_1, x0_2, x0_3, x0_4] to peak-center array
    allpeaks.append([fitParams[1], fitParams[4], fitParams[7], fitParams[10]])
    # append [sigma1, sigma2, sigma3,sigma4] to error array
    int_errors.append([fitParams[2], fitParams[5], fitParams[8], fitParams[11]])
    allparams.append(fitParams)

# set initial position in left plot
curr_pos = 0

# convert everything to numpy arrays
allplots_phis = np.asarray(allplots_phis)
allfits = np.asarray(allfits)
allpeaks = np.asarray(allpeaks)
int_errors = np.asarray(int_errors)
phis = np.asarray(phis)
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
    curr_pos = curr_pos % len(allplots_phis)

    ax.cla()
    # ax.plot(allplots_phis[curr_pos][0], allplots_phis[curr_pos][1])
    ax.scatter(allplots_phis[curr_pos][0], allplots_phis[curr_pos][1], marker="x")
    ax.plot(allfits[curr_pos][0], allfits[curr_pos][1], color='magenta', alpha=0.75)
    # if selected, plot individual gaussians
    if show_all_gauss:
        # if gauss mode is on, plot all individual gaussians resulting from the fit function
        g1 = gaussian(allfits[curr_pos][0], allparams[curr_pos][0], allparams[curr_pos][1],
                      allparams[curr_pos][2]) + offset
        ax.plot(allfits[curr_pos][0], g1, '--', color=c1)
        g2 = gaussian(allfits[curr_pos][0], allparams[curr_pos][3], allparams[curr_pos][4],
                      allparams[curr_pos][5]) + offset
        ax.plot(allfits[curr_pos][0], g2, '--', color='cyan')
        g3 = gaussian(allfits[curr_pos][0], allparams[curr_pos][6], allparams[curr_pos][7],
                      allparams[curr_pos][8]) + offset
        ax.plot(allfits[curr_pos][0], g3, '--', color=c3)
        g4 = gaussian(allfits[curr_pos][0], allparams[curr_pos][9], allparams[curr_pos][10],
                      allparams[curr_pos][11]) + offset
        ax.plot(allfits[curr_pos][0], g4, '--', color=c4)

    ax.set_title("$\\theta = 90°; \phi$ = " + str(phis[curr_pos]) + "°")
    ax.set(ylim=[0.986, 1.003])
    ax.set_ylabel("Normalized Intensity")
    ax.set_xlabel("RF [GHz]")
    fig.canvas.draw()


# ------- plotting time --------- #
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
ax.scatter(allplots_phis[curr_pos][0], allplots_phis[curr_pos][1], marker="x")
ax.plot(allfits[curr_pos][0], allfits[curr_pos][1], color='magenta', alpha=0.75)
if show_all_gauss:
    # if gauss mode is on, plot all individual gaussians resulting from the fit function
    g1 = gaussian(allfits[curr_pos][0], allparams[curr_pos][0], allparams[curr_pos][1], allparams[curr_pos][2]) + offset
    ax.plot(allfits[curr_pos][0], g1, '--', color=c1)
    g2 = gaussian(allfits[curr_pos][0], allparams[curr_pos][3], allparams[curr_pos][4], allparams[curr_pos][5]) + offset
    ax.plot(allfits[curr_pos][0], g2, '--', color='cyan')
    g3 = gaussian(allfits[curr_pos][0], allparams[curr_pos][6], allparams[curr_pos][7],
                  allparams[curr_pos][8]) + offset
    ax.plot(allfits[curr_pos][0], g3, '--', color=c3)
    g4 = gaussian(allfits[curr_pos][0], allparams[curr_pos][9], allparams[curr_pos][10],
                  allparams[curr_pos][11]) + offset
    ax.plot(allfits[curr_pos][0], g4, '--', color=c4)

ax.set_title("$\\theta = 90°; \phi$ = " + str(phis[curr_pos]) + "°")
ax.set(ylim=[0.986, 1.003])
ax.set_xlabel("RF [GHz]")
ax.set_ylabel("Normalized Intensity")
ax.tick_params(width=2.5, length=10, top=True, right=True, direction='in')

# ------------------------ figure 2 - heat map recreation from gaussians ------------------------ #
fig2 = plt.figure()
# plot peak positions with error bars
ax2 = fig2.add_subplot(111)

# extract individual peak information (location + errors(=sigma))
peak1_overphi = allpeaks[:, 0]
peak2_overphi = allpeaks[:, 1]
peak3_overphi = allpeaks[:, 2]
peak4_overphi = allpeaks[:, 3]
peak1_err = int_errors[:, 0]
peak2_err = int_errors[:, 1]
peak3_err = int_errors[:, 2]
peak4_err = int_errors[:, 3]

# plot peak position over phi
pk1 = ax2.errorbar(phis, peak1_overphi, yerr=peak1_err, ls='none', marker='x', capsize=3, color=c1, ms=8)
pk2 = ax2.errorbar(phis, peak2_overphi, yerr=peak2_err, ls='none', marker='4', capsize=3, color=c2, ms=10)
pk3 = ax2.errorbar(phis, peak3_overphi, yerr=peak3_err, ls='none', marker='4', capsize=3, color=c3, elinewidth=0.8,
                   ms=10, label=r"for the [1,1,1] NV axis")
pk4 = ax2.errorbar(phis, peak4_overphi, yerr=peak4_err, ls='none', marker='x', capsize=3, color=c4,
                   label=r"for the [$\bar{1},1,\bar{1}$] NV axis", ms=8)
# ---------------- some extra tweaking for the legend --------------------- #
# Create empty plot with blank marker containing extra label we need in the beginning
dummyplot, = ax2.plot([], [], ' ', label="Gaussian fitted intensity minima")
dummyplot_2, = ax2.plot([], [], ' ')
# remove error bars and pair corresponding legend entries, see the follwing links:
# --> https://stackoverflow.com/questions/15551561/error-bars-in-the-legend
# --> https://stackoverflow.com/questions/23698850/manually-set-color-of-points-in-legend
handles = [(dummyplot, dummyplot_2), (pk1[0], pk4[0]), (pk2[0], pk3[0])]
_, labels = ax2.get_legend_handles_labels()
# use them in the legend
ax2.legend(handles=handles, labels=labels, loc='center left', fontsize=15, markerscale=1.25,
           handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None)})

# --------- make figure pretty
ax2.set_ylabel("RF [GHz]")
ax2.set_xlabel("$\\phi$ [°]")
ax2.tick_params(width=2.5, length=10, top=True, right=True, direction='in')

plt.show()

# ----------------- save peak information to be used in simulation -------------------- #
if savemode:
    data_to_save = np.asarray([phis, peak1_overphi, peak1_err, peak2_overphi, peak2_err, peak3_overphi,
                               peak3_err, peak4_overphi, peak4_err])
    # creta or overwrite file and save peaks info
    with open('data/peaks_from_phi.npy', 'wb') as f:
        np.save(f, data_to_save)

# -------------- get B-projection angle theta' as a function of theta -------------- #
# restore units
peak3_overphi = peak3_overphi * 10 ** 9
peak4_overphi = peak4_overphi * 10 ** 9
peak3_err = peak3_err * 10 ** 9
peak4_err = peak4_err * 10 ** 9

# 1) [111] - axis
nv3_thetaPrime_best = thPrime(peak3_overphi)  # best value array
nv3_thetaPrime_error = np.sqrt(dthPrime_de(peak3_overphi) ** 2 * C.dE ** 2 + dthPrime_dd(peak3_overphi) ** 2 * C.dD ** 2
                               + dthPrime_dnu(peak3_overphi) ** 2 * peak3_err ** 2)
# 2) [-1 1 -1] - axis
nv4_thetaPrime_best = thPrime(peak4_overphi)  # best value array
# nv4_thetaPrime_error = np.sqrt(dthPrime_de(peak4_overphi) ** 2 * dE ** 2 + dthPrime_dd(peak4_overphi) ** 2 * dD ** 2
#                                + dthPrime_dnu(peak4_overphi) ** 2 * peak4_err ** 2)
nv4_thetaPrime_error = [
    np.sqrt(dthPrime_de(peak4_overphi[idx]) ** 2 * C.dE ** 2 + dthPrime_dd(peak4_overphi[idx]) ** 2 * C.dD ** 2
            + dthPrime_dnu(peak4_overphi[idx]) ** 2 * peak4_err[idx] ** 2) for idx in
    range(len(peak4_overphi))]
