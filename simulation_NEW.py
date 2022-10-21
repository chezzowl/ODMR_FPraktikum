# simulation and optimization used for our dataset

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import time
import itertools
import constants as C
from constants import SimParams as SP
from numba import njit


# ########################################## functions ################################################### #
@njit()
def rot_euler(a, b, g):
    """
    :param a: float, angle of third rotation (around z-axis) in radians
    :param b: float, angle of second rotation (around x-axis) in radians
    :param g: float, angle of first rotation (around z-axis) in radians
    :return: 3x3 numpy array, rotation matrix for a general rotation around the Euler angles a (=alpha), b (=beta),
    g (=gamma) defined as follows: rot_euler(a,b,g) = rotZ(a).rotX(b).rotZ(g)
    """
    return np.array([
        [
            np.cos(a) * np.cos(g) - np.cos(b) * np.sin(a) * np.sin(g),
            -np.cos(b) * np.cos(g) * np.sin(a) - np.cos(a) * np.sin(g),
            np.sin(a) * np.sin(b)
        ],
        [
            np.cos(g) * np.sin(a) + np.cos(a) * np.cos(b) * np.sin(g),
            np.cos(a) * np.cos(b) * np.cos(g) - np.sin(a) * np.sin(g),
            -np.cos(a) * np.sin(b)
        ],
        [
            np.sin(b) * np.sin(g),
            np.cos(g) * np.sin(b),
            np.cos(b)
        ]
    ])


@njit()
def Bvec(B_abs, theta, phi):
    """
    :param B_abs: absolute value of vector
    :param theta: polar angle in radians
    :param phi: azimuthal angle in radians
    :return: numpy array, B field vector in cartesian coordinates in the lab frame in terms of
    the varied angles theta and phi
    """
    return np.array([
        B_abs * np.cos(phi) * np.sin(theta),
        B_abs * np.sin(phi) * np.sin(theta),
        B_abs * np.cos(theta)
    ])


@njit()
def nvaxes_lab(a, b, g, nvrefs):
    """
    :param a: float, angle of third Euler rotation (around z-axis) in radians
    :param b: float, angle of second Euler rotation (around x-axis) in radians
    :param g: float, angle of first Euler rotation (around z-axis) in radians
    :return: list of size 4 containing numpy arrays of size 3, each row/array holds the orientation (vector) of the
    corresponding NV-axis after rotation by the Euler angles a,b,g
    """
    return [
        np.dot(rot_euler(a, b, g), nvrefs[0]),
        np.dot(rot_euler(a, b, g), nvrefs[1]),
        np.dot(rot_euler(a, b, g), nvrefs[2]),
        np.dot(rot_euler(a, b, g), nvrefs[3])
    ]


@njit()
def Bprojection_scalars(B_abs, theta, phi, a, b, g, nvrefs):
    """
    :param B_abs: B field amplitude [T]
    :param theta: polar angle in radians (for B field orientation)
    :param phi: azimuthal angle in radians (for B field orientation)
    :param a: see Euler angle definition
    :param b: see Euler angle definition
    :param g: see Euler angle definition
    :param nvrefs: list of size 4 containing arrays of size 3, holding all NV axes' reference orientation. In all
    instances, the already defined `nvaxes_ref` will be used for this. It is kept as an input argument because it is
    better practice when using numba.
    :return: list of size 4, where the i-th value corresponds to the projection of the B field vector
    onto the i-th NV axis (in the order specified in `nvaxes_ref`)
    """
    [nv1, nv2, nv3, nv4] = nvaxes_lab(a, b, g, nvrefs)  # get all 4 NV axis vectors
    res = [
        np.dot(Bvec(B_abs, theta, phi), nv1) / np.linalg.norm(nv1),
        np.dot(Bvec(B_abs, theta, phi), nv2) / np.linalg.norm(nv2),
        np.dot(Bvec(B_abs, theta, phi), nv3) / np.linalg.norm(nv3),
        np.dot(Bvec(B_abs, theta, phi), nv4) / np.linalg.norm(nv4)
    ]  # projections onto all 4 axes
    return res


@njit()
def Bprojection_vectors(B_abs, theta, phi, a, b, g, nvrefs):
    """
    :param B_abs: B field amplitude [T]
    :param theta: polar angle in radians (for B field orientation)
    :param phi: azimuthal angle in radians (for B field orientation)
    :param a: see Euler angle definition
    :param b: see Euler angle definition
    :param g: see Euler angle definition
    :param nvrefs: list of size 4 containing arrays of size 3, holding all NV axes' reference orientation. In all
    instances, the already defined `nvaxes_ref` will be used for this. It is kept as an input argument because it is
    better practice when using numba.
    :return: list of size 4 containing numpy arrays of size 3, where the i-th row/array corresponds to the
    PROJECTION VECTOR of the B field vector onto the i-th NV axis (in the order specified in `nvaxes_ref`) ..
    i.e. this function multiplies the result of `Bprojection_scalar` by the axes we are projecting upon
    """
    [nv1, nv2, nv3, nv4] = nvaxes_lab(a, b, g, nvrefs)
    return [
        nv1 * np.dot(Bvec(B_abs, theta, phi), nv1) / np.dot(nv1, nv1),
        nv2 * np.dot(Bvec(B_abs, theta, phi), nv2) / np.dot(nv2, nv2),
        nv3 * np.dot(Bvec(B_abs, theta, phi), nv3) / np.dot(nv3, nv3),
        nv4 * np.dot(Bvec(B_abs, theta, phi), nv4) / np.dot(nv4, nv4)
    ]


@njit()
def eigenfreqs(B_abs, theta, phi, d, e, a, b, g, nvrefs):
    """

    :param B_abs: B field amplitude [T]
    :param theta: polar angle in radians (for B field orientation)
    :param phi: azimuthal angle in radians (for B field orientation)
    :param d: zero field splitting [Hz]
    :param e: zero field asymmetry term [Hz]
    :param a: see Euler angle definition
    :param b: see Euler angle definition
    :param g: see Euler angle definition
    :param nvrefs: list of size 4 containing arrays of size 3, holding all NV axes' reference orientation. In all
    instances, the already defined `nvaxes_ref` will be used for this. It is kept as an input argument because it is
    better practice when using numba.
    :return: numpy array of size 1x8,
    the indices [0-3] correspond to the HIGHER frequency eigenvalue of the corresponding NV axis
    given the current crystal and B field orientations;
    the indices [4-7] correspond to the LOWER frequency eigenvalue of the corresponding NV axis
    given the current crystal and B field orientations;
    """
    # get projection VECTORS of B field onto nv axes
    proj_vecs = Bprojection_vectors(B_abs, theta, phi, a, b, g, nvrefs)
    # B*cos(theta') ** 2 value, i.e. squared projection value, for each axis
    [proj_nv1, proj_nv2, proj_nv3, proj_nv4] = [np.linalg.norm(proj_vecs[0]) ** 2, np.linalg.norm(proj_vecs[1]) ** 2,
                                                np.linalg.norm(proj_vecs[2]) ** 2, np.linalg.norm(proj_vecs[3]) ** 2]
    # square root terms for all axes
    [sq1, sq2, sq3, sq4] = [np.sqrt(e ** 2 + proj_nv1 * C.alpha_factor ** 2),
                            np.sqrt(e ** 2 + proj_nv2 * C.alpha_factor ** 2),
                            np.sqrt(e ** 2 + proj_nv3 * C.alpha_factor ** 2),
                            np.sqrt(e ** 2 + proj_nv4 * C.alpha_factor ** 2)]
    # return all 8 values as specified in the docs
    return np.array([d + sq1, d + sq2, d + sq3, d + sq4, d - sq1, d - sq2, d - sq3, d - sq4])


@njit()
def optimization(combis, thetaaxis, goal_theta_1, goal_theta_2, phiaxis, goal_phi_1, goal_phi_2, thetaref, phiref, nvrefs):
    """
    searches for an optimal (alpha, beta, gamma, B) combination given the measured data by minimizing the MSE
    between measured data (given as an input) and data being calculated using our model
    (alpha, beta, gamma are Euler angles, B denotes the B field magnitude)

    :param combis: list of lists; each list element holds 4 floats corresponding to : [alpha, beta, gamma, B]
    :param thetaaxis: theta values at which the measurements (theta sweep) were taken and at which
    the model will be evaluated
    :param goal_theta_1: measured data over `thetaaxis` for one NV axis, one of the goals for our model
    :param goal_theta_2: measured data over `thetaaxis` for second NV axis, one of the goals for our model
    :param phiaxis: phi values at which the measurements (phi sweep) were taken and at which the model will be evaluated
    :param goal_phi_1: measured data over `phiaxis` for one NV axis, one of the goals for our model
    :param goal_phi_2: measured data over `phiaxis` for second NV axis, one of the goals for our model
    :param thetaref: float, theta value at which the phi sweep was done, in RADIANS
    :param phiref: float, phi value at which the theta sweep was done, in RADIANS
    :param nvrefs: list of size 4 containing arrays of size 3, holding all NV axes' reference orientation. In all
    instances, the already defined `nvaxes_ref` will be used for this. It is kept as an input argument because it is
    better practice when using numba.
    :return: TODO
    """
    # track best values
    min_error = np.inf
    optimal_alpha = np.inf
    optimal_beta = np.inf
    optimal_gamma = np.inf
    optimal_B = np.inf
    ctr = 0  # counter for convenience
    # printing does not work for now ---> workaround: save in-between minima and print them out later
    local_minima = []
    # separately: save everything under a certain value
    # minval = 1.65 * 10 ** (-4)
    minval = 6.2 * 10 ** (-5)
    under_minval = []

    for a, b, g, Bf in combis:
        ctr = ctr + 1
        # matrix of dimension (len(phiaxis),8), each column holds the evolution of the corresponding eigenvalue
        # over a varying angle PHI
        op_phiresult_matrix = [eigenfreqs(Bf, thetaref, phival, C.D, C.E, a, b, g, nvrefs)
                               for phival in phiaxis]
        # we need first two columns in proper dimensions
        op_model1_phi = np.array([row[0] / (1 * 10 ** 9) for row in op_phiresult_matrix])
        op_model2_phi = np.array([row[1] / (1 * 10 ** 9) for row in op_phiresult_matrix])
        # matrix of dimension (len(thetaaxis),8), each column holds the evolution of the corresponding eigenvalue
        # over a varying angle THETA
        op_thetaresult_matrix = [eigenfreqs(Bf, thval, phiref, C.D, C.E, a, b, g, nvrefs)
                                 for thval in thetaaxis]
        # we need first two columns in proper dimensions
        op_model1_theta = np.array([row[0] / (1 * 10 ** 9) for row in op_thetaresult_matrix])
        op_model2_theta = np.array([row[1] / (1 * 10 ** 9) for row in op_thetaresult_matrix])

        # compute all mean squared errors
        op_mse_theta1 = np.square(np.subtract(op_model1_theta, goal_theta_1)).mean()
        op_mse_theta2 = np.square(np.subtract(op_model2_theta, goal_theta_2)).mean()
        op_mse_phi1 = np.square(np.subtract(op_model1_phi, goal_phi_1)).mean()
        op_mse_phi2 = np.square(np.subtract(op_model2_phi, goal_phi_2)).mean()

        # calc error and see if it is the smallest so far
        err = op_mse_theta1 + op_mse_theta2 + op_mse_phi1 + op_mse_phi2
        if err < min_error:
            min_error = err
            optimal_alpha = a
            optimal_beta = b
            optimal_gamma = g
            optimal_B = Bf
            local_minima.append([a, b, g, Bf, err])
        # save only values under reference minimum
        if err < minval:
            under_minval.append([a, b, g, Bf, err])
    # return results in specific order
    return optimal_alpha, optimal_beta, optimal_gamma, optimal_B, min_error, local_minima, under_minval


# ############################### GUI interaction ################################# #

# Define an action for modifying the lines when any slider's value changes
def sliders_on_changed(val):  # update eigenenergy plots
    # ----- plots with theta = Pi/2 and varying phi ----- #
    # matrix of dimension (len(phi_ax,8)), each column holds the evolution of the corresponding eigenvalue
    # over a varying angle phi
    phiresult_mat = np.array(
        [eigenfreqs(B_slider.val / (1 * 10 ** 3), SP.theta_ref, phival, d_slider.val * (1 * 10 ** 6),
                    e_slider.val * (1 * 10 ** 6), alpha_slider.val,
                    beta_slider.val, gamma_slider.val, nvaxes_ref) for phival in phi_ax])
    phi_line_up_1.set_ydata(phiresult_mat[:, 0] / (10 ** 9))
    phi_line_up_2.set_ydata(phiresult_mat[:, 1] / (10 ** 9))
    phi_line_up_3.set_ydata(phiresult_mat[:, 2] / (10 ** 9))
    phi_line_up_4.set_ydata(phiresult_mat[:, 3] / (10 ** 9))
    phi_line_down_1.set_ydata(phiresult_mat[:, 4] / (10 ** 9))
    phi_line_down_2.set_ydata(phiresult_mat[:, 5] / (10 ** 9))
    phi_line_down_3.set_ydata(phiresult_mat[:, 6] / (10 ** 9))
    phi_line_down_4.set_ydata(phiresult_mat[:, 7] / (10 ** 9))

    # ----- plots with phi = 0 and varying theta ----- #
    # matrix of dimension (len(theta_ax,8)), each column holds the evolution of the corresponding eigenvalue
    # over a varying angle theta
    thetaresult_mat = np.array(
        [eigenfreqs(B_slider.val / (1 * 10 ** 3), thetaval, SP.phi_ref, d_slider.val * (1 * 10 ** 6),
                    e_slider.val * (1 * 10 ** 6), alpha_slider.val,
                    beta_slider.val, gamma_slider.val, nvaxes_ref) for thetaval in theta_ax])
    theta_line_up_1.set_ydata(thetaresult_mat[:, 0] / (10 ** 9))
    theta_line_up_2.set_ydata(thetaresult_mat[:, 1] / (10 ** 9))
    theta_line_up_3.set_ydata(thetaresult_mat[:, 2] / (10 ** 9))
    theta_line_up_4.set_ydata(thetaresult_mat[:, 3] / (10 ** 9))
    theta_line_down_1.set_ydata(thetaresult_mat[:, 4] / (10 ** 9))
    theta_line_down_2.set_ydata(thetaresult_mat[:, 5] / (10 ** 9))
    theta_line_down_3.set_ydata(thetaresult_mat[:, 6] / (10 ** 9))
    theta_line_down_4.set_ydata(thetaresult_mat[:, 7] / (10 ** 9))

    # lastly, update projection information
    # [pr1, pr2, pr3, pr4] = np.abs(Bprojection_scalars(B0, SP.theta_ref, SP.phi_ref, alpha_slider.val,
    #                                               beta_slider.val, gamma_slider.val) * (np.sqrt(2) / 2) / B0)
    # SAME CORRECTION AS ABOVE .. left in case normalization is changed to not forget this!
    [pr1, pr2, pr3, pr4] = np.abs(
        np.array(Bprojection_scalars(B_slider.val / (1 * 10 ** 3), SP.theta_ref, SP.phi_ref, alpha_slider.val,
                                     beta_slider.val, gamma_slider.val, nvaxes_ref)) / (B_slider.val / (1 * 10 ** 3)))
    # lines drawn with the axis orientation information saved in the beginning
    line1.set_xdata([0, nvaxes_2D[0][0] * pr1])
    line1.set_ydata([0, nvaxes_2D[0][1] * pr1])
    line2.set_xdata([0, nvaxes_2D[1][0] * pr2])
    line2.set_ydata([0, nvaxes_2D[1][1] * pr2])
    line3.set_xdata([0, nvaxes_2D[2][0] * pr3])
    line3.set_ydata([0, nvaxes_2D[2][1] * pr3])
    line4.set_xdata([0, nvaxes_2D[3][0] * pr4])
    line4.set_ydata([0, nvaxes_2D[3][1] * pr4])
    fig.canvas.draw_idle()


# ################################################################################################ #


# --------------------- some 'global' presets we need later on  --------------------- #
# w.l.o.g. set NV axis directions in the reference position (= no rotation)
nvaxes_ref = np.array([
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0]
])
# 2D coordinates for arrow heads corresponding to the FNV axes above. used in the right plot in the simulation
nvaxes_2D = np.array([(1, 1), (-1, 1), (1, -1), (-1, -1)])  # for later convenience: list of TUPLES
# corresponding labels
nvaxes_labels = [r"[1,1,1]", r"[$\bar{1},1,\bar{1}$]", r"[$1,\bar{1},\bar{1}$]", r"[$\bar{1},\bar{1}$,1]"]

# --------------------- simulation starts here --------------------- #

# plotting preparation
axis_color = 'lightgoldenrodyellow'

# create all subplots
fig = plt.figure()
fig.set_size_inches(13, 8)
axPhi = fig.add_subplot(131)  # phi sweep axis
axTheta = fig.add_subplot(132)  # theta sweep axis
axProj = fig.add_subplot(133)  # axis for plotting the projections

# Adjust the subplots region to leave some space for the controls
fig.subplots_adjust(bottom=0.25)

# axes information
theta_ax = np.arange(SP.theta_min, SP.theta_max + SP.theta_step, SP.theta_step)
phi_ax = np.arange(SP.phi_min, SP.phi_max + SP.phi_step, SP.phi_step)

# draw arrows representing the projections in the right plot
arrowNV1 = mpatches.FancyArrowPatch((0, 0), nvaxes_2D[0], mutation_scale=30, ec='none', fc='bisque')
axProj.add_patch(arrowNV1)
arrowNV2 = mpatches.FancyArrowPatch((0, 0), nvaxes_2D[1], mutation_scale=30, ec='none', fc='bisque')
axProj.add_patch(arrowNV2)
arrowNV3 = mpatches.FancyArrowPatch((0, 0), nvaxes_2D[2], mutation_scale=30, ec='none', fc='bisque')
axProj.add_patch(arrowNV3)
arrowNV4 = mpatches.FancyArrowPatch((0, 0), nvaxes_2D[3], mutation_scale=30, ec='none', fc='bisque')
axProj.add_patch(arrowNV4)
# add axis labels
axProj.text(*nvaxes_2D[0], nvaxes_labels[0])
axProj.text(*nvaxes_2D[1], nvaxes_labels[1])
axProj.text(*nvaxes_2D[2], nvaxes_labels[2])
axProj.text(*nvaxes_2D[3], nvaxes_labels[3])

# ------------------- initial values ------------------- #
# initial values for Euler angles
# alpha = -1.437678
# beta = 2.058285
# gamma = 1.650667

alpha = 1.544173  # -1.437678
beta = 2.599939  # 2.058285
gamma = 1.437678  # 1.650667

# ---------------------------- Draw the initial projections ---------------------------- #
# these lines' orientation does not change, thus we always know their components depending on the current B-orientation
# we have: x-component = y-component = (sqrt(2) / 2) * normalized length (with maximum = 1)
# normalized with B0 such that full alignment corresponds to the line exactly filling the NV vector
# [p1, p2, p3, p4] = np.abs(Bprojection_scalars(B0, SP.theta_ref, SP.phi_ref, alpha, beta, gamma) * (np.sqrt(2) / 2) / B0 )
# CORRECTION: the sqrt is not necessary while the axis vectors are not normalized to 1 (they have l= 2/sqrt(2)) !!!
[p1, p2, p3, p4] = np.abs(np.array(Bprojection_scalars(C.B0, SP.theta_ref, SP.phi_ref, alpha, beta, gamma, nvaxes_ref))
                          / C.B0)
# [p1, p2, p3, p4] = np.abs(np.divide(Bprojection_scalars(B0, SP.theta_ref, SP.phi_ref, alpha, beta, gamma), B0))
# lines drawn with the axis orientation information saved in the beginning
line1 = Line2D([0, nvaxes_2D[0][0] * p1], [0, nvaxes_2D[0][1] * p1], marker='o', ms=4.5)  # [111] axis
line2 = Line2D([0, nvaxes_2D[1][0] * p2], [0, nvaxes_2D[1][1] * p2], marker='o', ms=4.5)  # [-11-1] axis
line3 = Line2D([0, nvaxes_2D[2][0] * p3], [0, nvaxes_2D[2][1] * p3], marker='o', ms=4.5)  # [1-1-1] axis
line4 = Line2D([0, nvaxes_2D[3][0] * p4], [0, nvaxes_2D[3][1] * p4], marker='o', ms=4.5)  # [-1-11] axis
axProj.add_line(line1)
axProj.add_line(line2)
axProj.add_line(line3)
axProj.add_line(line4)

# ---------------------------- Draw the initial eigenvalue plots ---------------------------- #
# --- and create line objects to be updated on slider changes

# matrix of dimension (len(phi_ax),8), each column holds the evolution of the corresponding eigenvalue
# over a varying angle PHI
phiresult_matrix = np.array(
    [eigenfreqs(C.B0, SP.theta_ref, phival, C.D, C.E, alpha, beta, gamma, nvaxes_ref)
     for phival in phi_ax])
# line objects for phi plot
[phi_line_up_1] = axPhi.plot(phi_ax, phiresult_matrix[:, 0] / (10 ** 9), linewidth=2, color='red')
[phi_line_up_2] = axPhi.plot(phi_ax, phiresult_matrix[:, 1] / (10 ** 9), linewidth=2, color='blue')
[phi_line_up_3] = axPhi.plot(phi_ax, phiresult_matrix[:, 2] / (10 ** 9), linewidth=2, color='green')
[phi_line_up_4] = axPhi.plot(phi_ax, phiresult_matrix[:, 3] / (10 ** 9), linewidth=2, color='magenta')
[phi_line_down_1] = axPhi.plot(phi_ax, phiresult_matrix[:, 4] / (10 ** 9), linewidth=2, color='red')
[phi_line_down_2] = axPhi.plot(phi_ax, phiresult_matrix[:, 5] / (10 ** 9), linewidth=2, color='blue')
[phi_line_down_3] = axPhi.plot(phi_ax, phiresult_matrix[:, 6] / (10 ** 9), linewidth=2, color='green')
[phi_line_down_4] = axPhi.plot(phi_ax, phiresult_matrix[:, 7] / (10 ** 9), linewidth=2, color='magenta')

# matrix of dimension (len(theta_ax),8), each column holds the evolution of the corresponding eigenvalue
# over a varying angle THETA
thetaresult_matrix = np.array(
    [eigenfreqs(C.B0, thetaval, SP.phi_ref, C.D, C.E, alpha, beta, gamma, nvaxes_ref)
     for thetaval in theta_ax])
# line objects for theta plot
[theta_line_up_1] = axTheta.plot(theta_ax, thetaresult_matrix[:, 0] / (10 ** 9), linewidth=2, color='red')
[theta_line_up_2] = axTheta.plot(theta_ax, thetaresult_matrix[:, 1] / (10 ** 9), linewidth=2, color='blue')
[theta_line_up_3] = axTheta.plot(theta_ax, thetaresult_matrix[:, 2] / (10 ** 9), linewidth=2, color='green')
[theta_line_up_4] = axTheta.plot(theta_ax, thetaresult_matrix[:, 3] / (10 ** 9), linewidth=2, color='magenta')
[theta_line_down_1] = axTheta.plot(theta_ax, thetaresult_matrix[:, 4] / (10 ** 9), linewidth=2, color='red')
[theta_line_down_2] = axTheta.plot(theta_ax, thetaresult_matrix[:, 5] / (10 ** 9), linewidth=2, color='blue')
[theta_line_down_3] = axTheta.plot(theta_ax, thetaresult_matrix[:, 6] / (10 ** 9), linewidth=2, color='green')
[theta_line_down_4] = axTheta.plot(theta_ax, thetaresult_matrix[:, 7] / (10 ** 9), linewidth=2, color='magenta')

# ----------------------- make plots pretty ----------------------- #
# --- set axis params
phititle = "$\\nu_{\pm}$ for $\\theta = %i°$ and $\\phi \in [%i°,\,%i°]$" % \
           (int(SP.theta_ref * 180 / np.pi), int(SP.phi_min * 180 / np.pi), int(SP.phi_max * 180 / np.pi))
axPhi.set(title=phititle, xlim=[SP.phi_min, SP.phi_max], ylim=[2.82, 2.92])
axPhi.set_ylabel("$\\nu$ [$10^9\,$Hz]", labelpad=-6)  # set labels separately to allow labelpadding
thetatitle = "$\\nu_{\pm}$ for $\\phi = %i°$ and $\\theta \in [%i°,\,%i°]$" % \
             (int(SP.phi_ref * 180 / np.pi), int(SP.theta_min * 180 / np.pi), int(SP.theta_max * 180 / np.pi))
axTheta.set(title=thetatitle, xlim=[SP.theta_min, SP.theta_max], ylim=[2.82, 2.92])
axTheta.set_ylabel("$\\nu$ [$10^9\,$Hz]", labelpad=-6)  # set labels separately to allow labelpadding
projtitle = "$\\vec{\mathrm{B}}$ on NV projections for $\\theta = %i°,\, \\phi = %i°$" % \
            (int(SP.theta_ref * 180 / np.pi), int(SP.phi_ref * 180 / np.pi))
axProj.set(title=projtitle, xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])

# # ---------- load and plot measurements ------------ #
with open(SP.phisweep_file, 'rb') as f:
    [m_phis, m_phipeaks1, m_phierr1, m_phipeaks2, m_phierr2, m_phipeaks3, m_phierr3, m_phipeaks4, m_phierr4] \
        = np.load(f)

with open(SP.thetasweep_file, 'rb') as f2:
    [m_thetas, m_thetapeaks1, m_thetaerr1, m_thetapeaks2, m_thetaerr2] = np.load(f2)
    # ###### for a different dataset .. should not be needed if instructions are followed exactly ####### #
    # [m_thetas, m_thetapeaks1, m_thetaerr1, m_thetapeaks2, m_thetaerr2, m_thetapeaks3, m_thetaerr3, m_thetapeaks4,
    #  m_thetaerr4] = np.load(f2)
    # ################################################################################################### #
# ------------------------------------------------ #

# deg to rad
m_thetas = m_thetas * (np.pi / 180)
m_phis = m_phis * (np.pi / 180)

axPhi.errorbar(m_phis, m_phipeaks1, yerr=m_phierr1, ls='none', marker='x', capsize=3, color='purple')
axPhi.errorbar(m_phis, m_phipeaks2, yerr=m_phierr2, ls='none', marker='4', capsize=3, color='olivedrab')
axPhi.errorbar(m_phis, m_phipeaks3, yerr=m_phierr3, ls='none', marker='4', capsize=3, color='tab:cyan', elinewidth=0.8,
               ms=10)
axPhi.errorbar(m_phis, m_phipeaks4, yerr=m_phierr4, ls='none', marker='x', capsize=3, color='goldenrod')

axTheta.errorbar(m_thetas, m_thetapeaks1, yerr=m_thetaerr1, ls='none', marker='x', capsize=3, color='purple')
axTheta.errorbar(m_thetas, m_thetapeaks2, yerr=m_thetaerr2, ls='none', marker='4', capsize=3, color='olivedrab')
# ################### again for the other dataset ################### #
# axTheta.errorbar(m_thetas, m_thetapeaks3, yerr=m_thetaerr3, ls='none', marker='4', capsize=3, color='tab:cyan',
#                  elinewidth=0.8, ms=10)
# axTheta.errorbar(m_thetas, m_thetapeaks4, yerr=m_thetaerr4, ls='none', marker='x', capsize=3, color='goldenrod')
# ################################################################### #

# --------- add additional axis area and draw sliders there ----------------- #
# --- Euler angles
alpha_slider_ax = fig.add_axes([0.02, 0.15, 0.38, 0.03], facecolor=axis_color)
alpha_slider = Slider(alpha_slider_ax, '$\\alpha$', - np.pi, np.pi, valinit=alpha, valfmt='%1.4f')
beta_slider_ax = fig.add_axes([0.02, 0.1, 0.38, 0.03], facecolor=axis_color)
beta_slider = Slider(beta_slider_ax, '$\\beta$', 0, np.pi, valinit=beta, valfmt='%1.4f')
gamma_slider_ax = fig.add_axes([0.02, 0.05, 0.38, 0.03], facecolor=axis_color)
gamma_slider = Slider(gamma_slider_ax, '$\\gamma$', - np.pi, np.pi, valinit=gamma, valfmt='%1.4f')
# --- D, E, B
d_slider_ax = fig.add_axes([0.55, 0.15, 0.35, 0.03], facecolor=axis_color)
d_slider = Slider(d_slider_ax, 'D $[10^6 \, \mathrm{Hz}]$', (C.D - C.dD) / (1 * 10 ** 6), (C.D + C.dD) / (1 * 10 ** 6),
                  valinit=C.D / (1 * 10 ** 6), valfmt='%1.2f')
e_slider_ax = fig.add_axes([0.55, 0.10, 0.35, 0.03], facecolor=axis_color)
e_slider = Slider(e_slider_ax, 'E $[10^6 \, \mathrm{Hz}]$', (C.E - C.dE) / (1 * 10 ** 6), (C.E + C.dE) / (1 * 10 ** 6),
                  valinit=C.E / (1 * 10 ** 6), valfmt='%1.3f')
B_slider_ax = fig.add_axes([0.55, 0.05, 0.35, 0.03], facecolor=axis_color)
B_slider = Slider(B_slider_ax, 'B [mT]', C.B0 * 10 ** 3, (C.B0 + 2 * C.B0 / 10) * 10 ** 3,
                  valinit=C.B0, valfmt='%1.3f')

# connect sliders to handler
alpha_slider.on_changed(sliders_on_changed)
beta_slider.on_changed(sliders_on_changed)
gamma_slider.on_changed(sliders_on_changed)
d_slider.on_changed(sliders_on_changed)
e_slider.on_changed(sliders_on_changed)
B_slider.on_changed(sliders_on_changed)

plt.show()

# ----------------------------- OPTIMIZATION PROCEDURE -----------------------------  #
# in the following, the variables 'steps_al_gam', 'min_al_gam', 'max_al_gam', 'min_beta', 'max_beta',
# 'Bmin', 'Bmax', 'steps_B' should be set to reasonable values depending on the dataset (and of course depending
# on the B-field during the experiment) ... the current values are the ones best for our experiment
if SP.optimize:
    # ---------  create parameter combinations ------------- #
    # ---- ranges and step size for Euler angles Euler angles
    # alpha and gamma
    steps_al_gam = 64  # step number for alpha and gamma which are in the range [-pi, pi]
    min_al_gam = -np.pi  # minimal value to look for
    max_al_gam = np.pi  # maximal value to look for
    # stepsize_al_gam = (max_al_gam - min_al_gam) / steps_al_gam  # resulting step size

    # beta
    steps_beta = int(steps_al_gam / 2)  # step number for beta in range [0, pi]
    min_beta = 0
    max_beta = np.pi
    # stepsize_beta = (max_beta - min_beta) / steps_beta  # resulting step size

    # create arrays to plot over
    op_alpha_arr = np.linspace(start=min_al_gam, stop=max_al_gam, num=steps_al_gam)
    op_beta_arr = np.linspace(start=min_beta, stop=max_beta, num=steps_beta)
    op_gamma_arr = np.linspace(start=min_al_gam, stop=max_al_gam, num=steps_al_gam)
    print("stepsize alpha/gamma: " + str(op_alpha_arr[1] - op_alpha_arr[0]))
    print("stepsize beta: " + str(op_beta_arr[1] - op_beta_arr[0]))
    # B field
    Bmin = C.B0
    Bmax = C.B0 + 0.1 * 10 ** (-3)  # [T]
    steps_B = 5
    op_B_arr = np.linspace(start=Bmin, stop=Bmax, num=steps_B)  # only one B field value for now
    allcombinations = np.array(list(itertools.product(op_alpha_arr, op_beta_arr, op_gamma_arr, op_B_arr)))
    print("Starting optimization with " + str(len(allcombinations)) + " parameter combinations")
    start = time.time()
    # ------------- optimization -------------- #
    (opt_alpha, opt_beta, opt_gamma, opt_B, opt_err, minima, relevant_vals) = \
        optimization(allcombinations, m_thetas, m_thetapeaks2, m_thetapeaks2, m_phis, m_phipeaks3, m_phipeaks4,
        SP.theta_ref, SP. phi_ref, nvaxes_ref)
    # ################### again for the other dataset ################### #
    # (opt_alpha, opt_beta, opt_gamma, opt_B, opt_err, minima, relevant_vals) = \
    #     optimization(allcombinations, m_thetas, m_thetapeaks3, m_thetapeaks4, m_phis, m_phipeaks3, m_phipeaks4,
    #                  SP.theta_ref, SP. phi_ref, nvaxes_ref)
    # ################################################################### #
    end = time.time()
    print("Optimization with {} parameter combinations took {}s.\n".format(len(allcombinations), end - start))
    print("RESULTS:\nOptimal alpha: {:.6f} \nOptimal beta:{:.6f} \nOptimal gamma:{:.6f} "
          "\nOptimal B:{:.6f}\n Min. error: {}".format(opt_alpha, opt_beta, opt_gamma, opt_B, opt_err))

    print("================================ \nRELEVANT MINIMA:")
    for roww in relevant_vals:
        print("alpha: {:.6f} \nbeta:{:.6f} \ngamma:{:.6f} "
              "\nB:{:.6f}\n Min. error: {}".format(roww[0], roww[1], roww[2], roww[3], roww[4]))


# -------- additional stuff for later on (maybe!) ---------- #
# # ------------------- add text boxes ------------------- #
# # TODO: hook this up properly .. when using the boxes below, change the slider box parameters to the following:
# #   [0.02, 0.15/0.10/0.05, 0.38, 0.03] for alpha,beta, gamma ... and [0.60, 0.15/0.10/0.05, 0.33, 0.03] for D, E, B
# alphabox_ax = fig.add_axes([0.475, 0.15, 0.05, 0.03])
# alphabox = TextBox(alphabox_ax, "$\\alpha$", label_pad=.03)
# betabox_ax = fig.add_axes([0.475, 0.10, 0.05, 0.03])
# betabox = TextBox(betabox_ax, "$\\beta$", label_pad=.03)
# gammabox_ax = fig.add_axes([0.475, 0.05, 0.05, 0.03])
# gammabox = TextBox(gammabox_ax, "$\\gamma$", label_pad=.03)
#
# # connect boxes to functionality - TODO
#
# # set initial box values
# alphabox.set_val(alpha)
# betabox.set_val(beta)
# gammabox.set_val(gamma)

# def import_dat_data():
#     # not the best style, but a global image storage will do the job for now
#     global IMG_DATA
#     global IMG_INFO
#     global IMG_WAVELENGTHS
#     global IMG_ENERGIES
#     global DAT_FILE_PATH
#     try:
#         root = tk.Tk(className='Import sif-file')
#         DAT_FILE_PATH = askopenfilename()
#         IMG_DATA, IMG_INFO = sif_reader.np_open(DAT_FILE_PATH)
#         IMG_DATA = np.asarray(IMG_DATA, dtype=float)
#         # num_img = np.shape(IMG_DATA)[0]
#         # for i in range(num_img):
#         #     IMG_DATA[i] = IMG_DATA[i]/np.max(IMG_DATA[i])
#         IMG_WAVELENGTHS = sif_reader.utils.extract_calibration(IMG_INFO)
#         IMG_ENERGIES = h * c * 10 ** 9 / IMG_WAVELENGTHS  # c*10**9 --> everything in nanometers
#         root.destroy()
#     except:
#         print('Wrong file')
#         root.destroy()