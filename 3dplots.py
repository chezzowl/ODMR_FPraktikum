import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, ndimage
from matplotlib import ticker, cm
from mpl_toolkits.mplot3d import Axes3D

plt.rc('font', size=8)  # controls default text sizes
plt.rc('axes', titlesize=20)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
plt.rc('figure', titlesize=8)  # fontsize of the figure title
plt.rcParams['axes.linewidth'] = 2

# -----------------------------------------------------------------------------

# range of measurements
m_start = 49
m_end = 67

# Hardcoded selection for which angle is varied within which range and which angle is constant
# Use one of the following options and comment the other !!

# 1)
# phi_start = 0
# phi_end = 90
# phi_step = 5
# phi = np.arange(phi_start, phi_end+phi_step, phi_step)
# theta = 90
# option = 1

# 2)
theta_start = 90
theta_end = 180
theta_step = 5
# the following is linked to the number of considered measurements
theta = np.arange(theta_start, theta_end + theta_step, theta_step)
phi = 0
option = 2

# -----------------------------------------------------------------------------

# total number of measurements = total number of sweeped angles = len(theta)
N = m_end - m_start + 1

# ------- x- and y-axes stay the same throughout the measurement --------------------- #
# ------- initialize them form any of the files we read (in this case, from the first)
testfile = 'measurements\\Measurement_' + str(m_start) + '.DAT'
df_test = pd.read_csv(testfile, skiprows=3, sep='\t')

# number and RF range of RF steps; same for all measurements
freqs_test = df_test.iloc[:, 0]
number_frequ_steps = len(freqs_test)
RF_range = np.linspace(min(freqs_test), max(freqs_test), number_frequ_steps) / 1e9  # corresp. to plot axis

# initialize final array
# x-axis holds RF frquencies, y-axis holds angle information (N = number of angles sweeped during measurement)
map_array = np.zeros([number_frequ_steps, N])

# ------------------------------------------------------------------------------------- #

# Going through considered measurements and fill up map array
for i in range(0, N):
    # read measurement into dataframe and load necessary information
    file = 'measurements\\Measurement_' + str(i + m_start) + '.DAT'
    df = pd.read_csv(file, skiprows=3, sep='\t')
    frequency = df.iloc[:, 0]
    intensity = df.iloc[:, 4]
    # fill i-th column with the intensity array of the current measurement
    map_array[:, i] = intensity

# set varied and fixed angle depending on earlier choice
# this is curcial since 'varied_angle' corresponds to one of the axes plotted below!
if option == 1:
    varied_angle = phi
    fixed_angle = theta
else:
    varied_angle = theta
    fixed_angle = phi

# normalize intensity values
map_array = map_array / map_array.max()

# -- ASIDE: first and second derivative filtering
laplace2D = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]])

# properly set up gauss filter with window size 5
sigma = 2
windowsize = 5
# properly set truncate variable
trunc = t = (((windowsize - 1) / 2) - 0.5) / sigma

gauss = ndimage.filters.gaussian_filter(map_array, sigma=sigma, truncate=trunc, order=(2, 2))
second_der = signal.convolve2d(map_array, laplace2D, boundary='fill', mode='same')  # not used for now

# -----------------------------------------------------------------------------

# ----------------------- create 3d plot ----------------------- #

# fig, ax = plt.subplots(figsize=(20, 12))
# im = ax.contourf(varied_angle, RF_range, map_array, 200, cmap='inferno')
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
# ax3 = fig.add_subplot(133, projection='3d')

X, Y = np.meshgrid(varied_angle, RF_range)
im = ax.plot_surface(X, Y, map_array, cmap="plasma")
im2 = ax2.plot_surface(X, Y, gauss, cmap="plasma")
# im3 = ax3.plot_surface(X, Y, second_der, cmap="plasma")
if option == 1:
    ax.set_xlabel('$\\varphi$ [°]')
else:
    ax.set_xlabel('$\\theta$ [°]')
ax.set_ylabel('Microwave frequency [GHz]')
ax.set_ylim(2.8, 2.95)

plt.show()
