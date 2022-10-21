import numpy as np

"""
constants used throughout the scripts
"""
# values for D and E obtained from zero-field splitting
# IMPORTANT: these values are used in the optimization procedure performed in the simulation file
D = 2.8690 * 10 ** 9  # [Hz]
dD = 0.0015 * 10 ** 9  # [Hz]
E = 3.1 * 10 ** 6  # [Hz]
dE = 1.5 * 10 ** 6  # [Hz]

ge = 2.00231930436256
muB = 9.2740100783 * 10 ** (-24)  # [J/T]
h = 6.62607015 * 10 ** (-34)  # [J/Hz]
B0 = 1 * 10 ** (-3)  # [T]
alpha_factor = ge * muB / h  # [Hz/T], combined factor of natural constants used as an abbreviation
# --- for another dataset --- #
# D = 2.868954 * 10 ** 9  # [Hz]
# dD = 0.0015 * 10 ** 9  # [Hz]
# E = 5.4 * 10 ** 6  # [Hz]
# dE = 1.5 * 10 ** 6  # [Hz]


class SimParams:
    """
    storage for parameters needed for the simulation
    """
    # reference values for theta and phi
    # these are the fixed values at which the other angle was sweeped
    # (e.g: we sweeped theta for phi=0° and phi for theta=90°)
    # Note: adjust these according to your measurement.
    theta_ref = np.pi / 2
    phi_ref = 0

    # range for the axes over which the eigenvalues are plotted
    # --> should be big enough to include the whole measurement range
    # in RADIANS
    # Note: adjust these according to your measurement.
    theta_min = np.pi / 2
    theta_max = np.pi
    theta_step = (np.pi / 2) / 100  # resolution
    phi_min = 0.0
    phi_max = np.pi / 2
    phi_step = (np.pi / 2) / 100  # resolution

    optimize = 0  # if set to true (= non-zero), optimization procedure starts automatically after closing GUI
    # Attention: this is NOT DONE in the 'mainSim' file for now!

    # numpy files with measurement data
    # By measurement data, we mean the following two measurements:
    # 1) positions of the intensity minima during the phi-angle sweep at theta = 'theta_ref' (set above!)
    # 2) positions of the intensity minima during the theta-angle sweep at phi = 'phi_ref' (set above!)
    # Each of the two files must have the following structure:
    # [x - axis, measurement 1, errors measurement 1, measurement 2, errors measurement 2, ...]
    # If there are no error arrays as of now, just save zero-arrays as errors into the file.
    #
    # Note: The file paths are RELATIVE, e.g. these files should be stored in the 'data' directory of this project.
    # Note: adjust these according to your measurement.
    phisweep_file = 'data/peaks_from_phi.npy'
    thetasweep_file = 'data/peaks_from_theta.npy'
