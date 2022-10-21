# ATTENTION: NOT WORKING FOR NOW
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

# ------- preliminary plot settings for later ---------- #
plt.rc('font', size=10)  # controls default text sizes
plt.rc('axes', titlesize=35)  # fontsize of the axes title
plt.rc('axes', labelsize=35)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=28)  # fontsize of the tick labels
plt.rc('ytick', labelsize=28)  # fontsize of the tick labels
plt.rc('figure', titlesize=7)  # fontsize of the figure title
plt.rcParams['axes.linewidth'] = 2

# -------- start procedure ------- #

print("Choose angle to be varied ...")
response = None
while response not in {"Phi", "phi", "Theta", "theta"}:
    response = input("Please enter phi or theta: ")

# ---- default choice is phi, otherwise variables are varied angle is theta ---- #

# start and end value of measurement:
m_start = 30
m_end = 48

angle_start = 0
angle_end = 90

phi_start = 0
phi_end = 90
phi_step = 5
phi = np.arange(phi_start, phi_end + phi_step, phi_step)
theta = 90
option = 0

if response == "theta" or response == "Theta":
    m_start = 49
    m_end = 67
    theta_start = 90
    theta_end = 180
    theta_step = 5
    theta = np.arange(theta_start, theta_end + theta_step, theta_step)
    phi = 0
    option = 1

# ---- parameters based on choice are set, now evaluate ---- #
# number of considered measurements for color map
N = m_end - m_start + 1

# first measurement as test file to initialize array
testfile = 'measurements\\Measurement_' + str(m_start) + '.DAT'
df_test = pd.read_csv(testfile, skiprows=3, sep='\t')

# number of RF steps
number_frequ_steps = len(df_test.iloc[:, 0])

# initialize map array that contains the intensity with respect to the 
# RF along a row and with respect to the angle along a column
map_array = np.zeros([number_frequ_steps, N])

# Going through considered measurements and fill up map array
for i in range(0, N):
    file = 'measurements\\Measurement_' + str(i + m_start) + '.DAT'
    df = pd.read_csv(file, skiprows=3, sep='\t')
    frequency = df.iloc[:, 0]
    intensity = df.iloc[:, 4]
    map_array[:, i] = intensity

# Finding the range of varied angle and value of fixed angle on basis of 
# the option taken above
if option:  # theta is varied
    varied_angle = theta
    fixed_angle = phi
else:  # phi is varied
    varied_angle = phi
    fixed_angle = theta

# range of RF in GHz
RF_range = np.linspace(min(frequency), max(frequency), len(frequency)) / 1e9

# lastly normalize the intensity values of map array
map_array = map_array / map_array.max()

# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(20, 12))
im = ax.contourf(varied_angle, RF_range, map_array, 200, cmap='inferno')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Normalized Intensity', rotation=270, labelpad=65)
if option:
    ax.set_xlabel('$\\theta$ [°]')
else:
    ax.set_xlabel('$\\phi$ [°]')
ax.set_ylabel('RF [GHz]')
ax.set_ylim(2.8, 2.95)
tick_locator = ticker.MaxNLocator(nbins=5)
cbar.locator = tick_locator
cbar.update_ticks()
# plt.locator_params(axis='y', nbins=4)


plt.show()
