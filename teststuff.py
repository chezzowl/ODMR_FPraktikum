import numpy as np
import itertools
import constants as C
from numba import njit

# --------------------- some 'global' presets we need later on  --------------------- #
# reference position of NV axis vectors, each row holds the orientation (vector) of one NV axis
nvaxes_ref = np.array([
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0]
])

# reference theta and phi values needed later on
theta_ref = np.pi / 2
phi_ref = 0


# ########################################## functions ################################################### #

@njit()
def rot_euler(a, b, g):
    """
    :param a: float, angle of third rotation (around z-axis) in radians
    :param b: float, angle of second rotation (around x-axis) in radians
    :param g: float, angle of first rotation (around z-axis) in radians
    :return: rotation matrix for a general rotation around the Euler angles a (=alpha), b (=beta), g (=gamma) defined
        as follows: rot_euler(a,b,g) = rotZ(a).rotX(b).rotZ(g)
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
    :return: B field vector in cartesian coordinates in the lab frame in terms of the varied angles theta and phi
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
    :return: numpy array of dimensions 3x4, each row holds the orientation (vector) of the corresponding NV-axis
        after rotation by the Euler angles a,b,g
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
    :return: numpy array of dimensions 1x4, where the i-th value corresponds to the projection of the B field vector
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
    :return: numpy array of dimensions 3x4, where the i-th row corresponds to the PROJECTION VECTOR of the
    B field vector onto the i-th NV axis (in the order specified in `nvaxes_ref`) ..
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


# ################################################################################################ #

# @njit()
# def smallfunc(al, bet, gam, Bfield, phiaxis):
#     test = eigenfreqs(Bfield, theta_ref, phiaxis[0], C.D, C.E, al, bet, gam)
#     print(test)
#     # op_phiresult_matrix = [eigenfreqs(Bfield, theta_ref, phival, C.D, C.E, al, bet, gam)
#     #                        for phival in phiaxis]
#     # print(op_phiresult_matrix)


@njit()
def optimization(combis, thetaaxis, goal_theta, phiaxis, goal_phi_1, goal_phi_2, nvrefs):
    """
    searches for an optimal (alpha, beta, gamma, B) combination given the measured data by minimizing the MSE
    between measured data (given as an input) and data being calculated using our model
    (alpha, beta, gamma are Euler angles, B denotes the B field magnitude)
    :param combis: list of lists; each list element holds 4 floats corresponding to : [alpha, beta, gamma, B]
    :param thetaaxis: theta values at which the measurements (theta sweep) were taken and at which
    the model will be evaluated
    :param goal_theta: measured data over `thetaaxis` for one NV axis, one of the goals for our model
    :param phiaxis: phi values at which the measurements (phi sweep) were taken and at which the model will be evaluated
    :param goal_phi_1: measured data over `phiaxis` for one NV axis, one of the goals for our model
    :param goal_phi_2: measured data over `phiaxis` for second NV axis, one of the goals for our model
    """
    # track best values
    min_error = np.inf
    optimal_alpha = np.inf
    optimal_beta = np.inf
    optimal_gamma = np.inf
    ctr = 0  # counter for convenience
    # printing does not work for now ---> workaround: save in-between minima and print them out later
    local_minima = []

    for a, b, g, Bf in combis:
        ctr = ctr + 1
        # matrix of dimension (len(phiaxis),8), each column holds the evolution of the corresponding eigenvalue
        # over a varying angle PHI
        op_phiresult_matrix = [eigenfreqs(Bf, theta_ref, phival, C.D, C.E, a, b, g, nvrefs)
                               for phival in phiaxis]
        # we need first two columns in proper dimensions
        op_model1_phi = np.array([row[0] / (1 * 10 ** 9) for row in op_phiresult_matrix])
        op_model2_phi = np.array([row[1] / (1 * 10 ** 9) for row in op_phiresult_matrix])
        # matrix of dimension (len(thetaaxis),8), each column holds the evolution of the corresponding eigenvalue
        # over a varying angle THETA
        op_thetaresult_matrix = [eigenfreqs(Bf, thval, phi_ref, C.D, C.E, a, b, g, nvrefs)
                                 for thval in thetaaxis]
        # we need first two columns in proper dimensions
        op_model1_theta = np.array([row[0] / (1 * 10 ** 9) for row in op_thetaresult_matrix])
        op_model2_theta = np.array([row[1] / (1 * 10 ** 9) for row in op_thetaresult_matrix])

        # compute all mean squared errors
        op_mse_theta1 = np.square(np.subtract(op_model1_theta, goal_theta)).mean()
        op_mse_theta2 = np.square(np.subtract(op_model2_theta, goal_theta)).mean()
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
            local_minima.append([a, b, g, Bf, min_error])
            # print('Counter: ' + str(ctr) + '\ncurrent alpha: ' + str(optimal_alpha) + '\ncurrent beta: '
            #       + str(optimal_beta) + '\ncurrent gamma: ' + str(optimal_gamma) + '\ncurrent B: ' + str(optimal_B)
            #       + '\ncurrent error: ' + str(min_error))
            # print(f"Counter: {ctr}\ncurrent alpha: {optimal_alpha:.2f} \ncurrent beta:{optimal_beta:.2f} \n"
            #       f"current gamma:{optimal_gamma:.2f}\ncurrent B:{optimal_B:.6f}\n Min. error: {min_error:.9f}")
    # return results in specific order
    return optimal_alpha, optimal_beta, optimal_gamma, optimal_B, min_error, local_minima


if __name__ == "__main__":
    # ########################### find optimal alpha, beta, gamma, B ########################### #

    # ---------- load measurements from files ------------ #
    with open('data/peaks_from_phi.npy', 'rb') as f:
        [m_phis, m_phipeaks1, m_phierr1, m_phipeaks2, m_phierr2, m_phipeaks3, m_phierr3, m_phipeaks4, m_phierr4] \
            = np.load(f)

    with open('data/peaks_from_theta.npy', 'rb') as f2:
        [m_thetas, m_thetapeaks1, m_thetaerr1, m_thetapeaks2, m_thetaerr2] = np.load(f2)

    # convert to radians
    m_phis = m_phis * (np.pi / 180)
    m_thetas = m_thetas * (np.pi / 180)

    # ---------  create parameter combinations ------------- #
    # Euler angles
    steps_al_gam = 10  # step number for alpha and gamma which are in the range [-pi, pi]
    min_al_gam = -np.pi
    max_al_gam = np.pi
    # stepsize_al_gam = (max_al_gam - min_al_gam) / steps_al_gam  # resulting step size
    steps_beta = int(steps_al_gam / 2)  # step number for beta in range [0, pi]
    min_beta = 0
    max_beta = np.pi
    # stepsize_beta = (max_beta - min_beta) / steps_beta  # resulting step size
    op_alpha_arr = np.linspace(start=min_al_gam, stop=max_al_gam, num=steps_al_gam).astype(np.float64)
    op_beta_arr = np.linspace(start=min_beta, stop=max_beta, num=steps_beta).astype(np.float64)
    op_gamma_arr = np.linspace(start=min_al_gam, stop=max_al_gam, num=steps_al_gam).astype(np.float64)
    op_B_arr = [1.1 * 10 ** (-3)]  # only one B field value for now
    allcombinations = np.array(list(itertools.product(op_alpha_arr, op_beta_arr, op_gamma_arr, op_B_arr)))
    print(len(allcombinations))
    # ------------- the procedure -------------- #
    (opt_alpha, opt_beta, opt_gamma, opt_B, opt_err, minima) = \
        optimization(allcombinations, m_thetas, m_thetapeaks2, m_phis, m_phipeaks3, m_phipeaks4, nvaxes_ref)
    print("Optimal alpha: {:.6f} \nOptimal beta:{:.6f} \nOptimal gamma:{:.6f} "
          "\nOptimal B:{:.6f}\n Min. error: {}".format(opt_alpha, opt_beta, opt_gamma, opt_B, opt_err))
    print(minima)
    # smallfunc(op_alpha_arr[0], op_beta_arr[0], op_gamma_arr[0], op_B_arr[0],m_phis)
