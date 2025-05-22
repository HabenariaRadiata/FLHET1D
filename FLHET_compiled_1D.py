import numpy as np
import math
import scipy.constants as phy_const
import os
import pickle
import sys
from numba import njit
import time as ttime
import glob
import scipy.interpolate as interpolate
from scipy import integrate

from modules.simu_params import SimuParameters

#########################################################
# We solve for a system of equations written as
# dU/dt + dF/dx = S
# with a Finite Volume Scheme.
#
# The conservative variables are
# U = [rhog, rhoi, rhoUi, 3/2 ne*e*Te],
# F = [rhog*Vg, rhoUi, rhoUi*Ui + ne*e*Te, 5/2 ne*e*Te*Ue].
#
# We use the following primitive variables
# P = [ng, ni,  ui,  Te, ve_x, ve_y]
#
# At the boundaries we impose
# Inlet:
#       ng = MDOT/(M*A0*VG)*M
#       ui = -u_bohm
# Outlet:
#       Te = Te_Cath
#       ve = 0
#
# The user can change the PHYSICAL PARAMETERS
# or the NUMERICAL PARAMETERS
#
#
# This script FLHET_compiled.py has a few functions compiled with numba.
# It is approximately 2.3 times faster than its not compiled counterpart:
# the LPP1D.py script.
##########################################################


tttime_start = ttime.time()

configfile = sys.argv[1]

msp = SimuParameters(configfile)

##### Renames th variables for more clarity #####
Resultsdir = msp.Results
MDOT = msp.MDOT
Mi = msp.Mi
A0 = msp.A0
VG = msp.VG
NI0 = msp.NI0
TE0 = msp.TE0
ESTAR = msp.ESTAR
wall_inter_type = msp.wall_inter_type
R1 = msp.R1
R2 = msp.R2
LX = msp.LX
LTHR = msp.LTHR
KEL = msp.KEL
TIMESCHEME = msp.TIMESCHEME
TIMEFINAL = msp.TIMEFINAL
SAVERATE = msp.SAVERATE
boolIonColl = msp.boolIonColl
boolSizImposed = msp.boolSizImposed
Eion = msp.Eion
gamma_i = msp.gamma_i
Te_inj = msp.Te_inj
Rext = msp.Rext
Te_Cath = msp.Te_Cath
boolPressureDiv = (
    msp.boolPressureDiv
)  # is dp/dx * u_e  accounted in the energy equation
CFL = msp.CFL
HEATFLUX = msp.HEATFLUX
IMPlICIT = msp.IMPlICIT
boolCircuit = msp.Circuit
V = msp.V0
thomas_BM_testcase = msp.thomas_BM_testcase
empirical_term = msp.empirical_term
empirical_term_path = msp.empirical_term_path
T_min = 1.0

# Set global variables
me = phy_const.m_e


if os.path.exists(Resultsdir):
    # delete current data in location:
    print("!!!!! Warning !!!!!")
    print(
        "All Macroscopic*.pkl files in the "
        + Resultsdir
        + " location will be deleted to welcome new data."
    )
    print("Do you want to continue? (y/n)")
    answer = input()

    bool_rewrite = False
    while bool_rewrite == False:
        if answer == "y":
            list_of_res_files = glob.glob(Resultsdir + "/Data/Macroscopic*.pkl")
            for filenametemp in list_of_res_files:
                os.remove(filenametemp)
            if len(list_of_res_files) > 0:
                print(
                    "Warning: all Macroscopic*.pkl files in the "
                    + Resultsdir
                    + " location were deleted to welcome new data."
                )
            bool_rewrite = True
        if answer == "n":
            print("The program will stop now.")
            sys.exit()

if not os.path.exists(Resultsdir):
    os.makedirs(Resultsdir)
ResultsFigs = Resultsdir + "/Figs"
if not os.path.exists(ResultsFigs):
    os.makedirs(ResultsFigs)
ResultsData = Resultsdir + "/Data"
if not os.path.exists(ResultsData):
    os.makedirs(ResultsData)

msp.save_config_file("Configuration.cfg")

Delta_t = 1.0  # Initialization of Delta_t (do not change)

##########################################################
#           Allocation of large vectors                  #
##########################################################

x_mesh, x_center, Delta_x, x_center_extended, Delta_x_extended = (
    msp.return_tiled_domain()
)
NBPOINTS = np.shape(x_center)[0]


def compute_B_array():
    """
    Compute the magnetic field array Barr.
    """
    BMAX = msp.BMAX
    B0 = msp.B0
    BLX = msp.BLX
    LTHR = msp.LTHR
    LB1 = msp.LB1
    LB2 = msp.LB2

    if msp.BTYPE == "CharoyBenchmark":

        a1 = (BMAX - B0) / (1 - math.exp(-(LTHR**2) / (2 * LB1**2)))
        a2 = (BMAX - BLX) / (1 - math.exp(-((LX - LTHR) ** 2) / (2 * LB2**2)))
        b1 = BMAX - a1
        b2 = BMAX - a2
        Barr1 = a1 * np.exp(-((x_center - LTHR) ** 2) / (2 * LB1**2)) + b1
        Barr2 = (
            a2 * np.exp(-((x_center - LTHR) ** 2) / (2 * LB2**2)) + b2
        )  # Magnetic field outside the thruster

        Barr = np.where(x_center <= LTHR, Barr1, Barr2)

    else:
        Barr = BMAX * np.exp(
            -(((x_center - LTHR) / LB1) ** 2.0)
        )  # Magnetic field within the thruster
        Barr = np.where(
            x_center < LTHR, Barr, BMAX * np.exp(-(((x_center - LTHR) / LB2) ** 2.0))
        )  # Magnetic field outside the thruster

    return Barr


Barr = compute_B_array()
Barr_extended = np.concatenate([[Barr[0]], Barr, [Barr[-1]]])
wce = phy_const.e * Barr / me  # electron cyclotron frequency

alpha_B1, alpha_B2 = msp.extract_anom_coeffs()


def compute_alphaB_array():
    """
    Compute the anomalous transport coefficient array alpha_B.
    """
    alpha_B = (
        np.ones(NBPOINTS) * alpha_B1
    )  # Anomalous transport coefficient inside the thruster
    alpha_B = np.where(
        x_center < msp.LTHR, alpha_B, alpha_B2
    )  # Anomalous transport coefficient in the plume
    alpha_B_smooth = np.copy(alpha_B)

    # smooth between alpha_B1 and alpha_B2
    nsmooth_o2 = msp.NBPOINTS_INIT // 20
    for index in range(nsmooth_o2, NBPOINTS - (nsmooth_o2 + 1)):
        alpha_B_smooth[index] = np.mean(
            alpha_B[index - nsmooth_o2 : index + (nsmooth_o2 + 1)]
        )

    return alpha_B_smooth


alpha_B = compute_alphaB_array()
alpha_B_extended = np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]])


def linear_extrapolation_multi(vec, num_points=3):
    """
    Linearly extrapolate the first and last few points of a vector.
    """
    # Use at least two points for linear extrapolation
    if len(vec) < 2:
        raise ValueError("Vector must have at least two elements for extrapolation.")

    # Ensure num_points is not greater than the vector length
    num_points = min(num_points, len(vec) - 1)

    # Fit a line (degree 1 polynomial) to the first few points
    start_fit = np.polyfit(range(num_points), vec[:num_points], 1)
    # Predict the value before the first point using the line
    start_extrapolated_value = np.polyval(
        start_fit, -1
    )  # x = -1 for extrapolation one step back

    # Fit a line to the last few points
    end_fit = np.polyfit(range(len(vec) - num_points, len(vec)), vec[-num_points:], 1)
    # Predict the value after the last point using the line
    end_extrapolated_value = np.polyval(
        end_fit, len(vec)
    )  # x = len(vec) for extrapolation one step forward

    # Add the extrapolated values to the vector
    extended_vec = np.insert(
        vec, 0, start_extrapolated_value
    )  # Insert at the beginning
    extended_vec = np.append(extended_vec, end_extrapolated_value)  # Append at the end

    return extended_vec

print("empirical_term: ", empirical_term)
if empirical_term:
    try:
        empirical_term = np.loadtxt(empirical_term_path)
        # empirical_term = np.loadtxt('no_Rei.txt')
        empirical_term_interp_y = np.interp(
            x_center, empirical_term[:, 0] / 100, empirical_term[:, 1]
        )
        print("Empirical term y loaded.")
        empirical_term_interp_x = np.interp(
            x_center, empirical_term[:, 0] / 100, empirical_term[:, 2]
        )
        print("Empirical term x loaded.")
        tau_xy_temp = np.interp(
            x_center, empirical_term[:, 0] / 100, empirical_term[:, 3]
        )
        tau_xy = linear_extrapolation_multi(tau_xy_temp, 2)
        print("tau_xy loaded")
        heat_flux_temp = np.interp(
            x_center, empirical_term[:, 0] / 100, empirical_term[:, 4]
        )
        heat_flux = linear_extrapolation_multi(heat_flux_temp, 2) * 0
        if sum(heat_flux) == 0:
            print("No heat flux")
        else:
            print("heatflux loaded")
    except:
        print("No empirical term file found.")
else:
    tau_xy = np.zeros(NBPOINTS+2)
    heat_flux = np.zeros(NBPOINTS+2)
    empirical_term_interp_y = np.zeros(NBPOINTS)
    empirical_term_interp_x = np.zeros(NBPOINTS)

##### Save invariant data #####
pickle.dump(
    [Barr, x_mesh, x_center, alpha_B],
    open(ResultsData + "/MacroscopicUnvariants.pkl", "wb"),
)

##### Allocation of vectors #####
P = np.ones(
    (6, NBPOINTS)
)  # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
U = np.ones((5, NBPOINTS))  # Conservative vars U = [rhog, rhoi, rhoUi, 3/2 ne*e*Te]
S = np.ones((5, NBPOINTS))  # Source Term
Efield = np.zeros(NBPOINTS)
F_cell = np.ones(
    (5, NBPOINTS + 2)
)  # Flux at the cell center. We include the Flux of the Ghost cells
F_interf = np.ones((5, NBPOINTS + 1))  # Flux at the interface
U_LeftGhost = np.ones((5, 1))  # Ghost cell on the left
P_LeftGhost = np.ones((6, 1))  # Ghost cell on the left
U_RightGhost = np.ones((5, 1))  # Ghost cell on the right
P_RightGhost = np.ones((6, 1))  # Ghost cell on the right

if msp.START_FROM_INPUT:
    list_of_pkls = glob.glob(msp.INPUT_DIR + "/MacroscopicVars*.pkl")
    INPUT_FILE = list_of_pkls[-1]
    print("Simulation starts from the profiles stored in ", INPUT_FILE)

    with open(
        INPUT_FILE, "rb"
    ) as f:  # !!! Caution !!!: outdated, because currently data are not saved like this. May malfunction for input file recently added.
        [
            t_INIT,
            P_INIT,
            U_INIT,
            P_LeftGhost_INIT,
            P_RightGhost_INIT,
            J_INIT,
            V_INIT,
            B_INIT,
            x_center_INIT,
        ] = pickle.load(f)

    NBPOINTS_initialField = P_INIT.shape[1]
    # Delta_x_initialField  = LX/NBPOINTS_initialField
    x_mesh_initialField = np.zeros(
        NBPOINTS_initialField + 1, dtype=float
    )  # Mesh in the interface
    x_mesh_initialField[1:-1] = 0.5 * (x_center_INIT[:-1] + x_center_INIT[1:])
    x_mesh_initialField[-1] = LX
    # x_center_initialField = np.linspace(Delta_x_initialField, LX - Delta_x_initialField, NBPOINTS_initialField)     # Mesh in the center of cell

    # interpolation of the initial profiles on the current mesh
    P0_INTERP = interpolate.interp1d(
        x_center_INIT,
        P_INIT[0, :],
        fill_value=(P_INIT[0, 0], P_INIT[0, -1]),
        bounds_error=False,
    )
    P1_INTERP = interpolate.interp1d(
        x_center_INIT,
        P_INIT[1, :],
        fill_value=(P_INIT[1, 0], P_INIT[1, -1]),
        bounds_error=False,
    )
    P2_INTERP = interpolate.interp1d(
        x_center_INIT,
        P_INIT[2, :],
        fill_value=(P_INIT[2, 0], P_INIT[2, -1]),
        bounds_error=False,
    )
    P3_INTERP = interpolate.interp1d(
        x_center_INIT,
        P_INIT[3, :],
        fill_value=(P_INIT[3, 0], P_INIT[3, -1]),
        bounds_error=False,
    )
    P4_INTERP = interpolate.interp1d(
        x_center_INIT,
        P_INIT[4, :],
        fill_value=(P_INIT[4, 0], P_INIT[4, -1]),
        bounds_error=False,
    )
    P5_INTERP = interpolate.interp1d(
        x_center_INIT,
        P_INIT[5, :],
        fill_value=(P_INIT[5, 0], P_INIT[5, -1]),
        bounds_error=False,
    )

    # We initialize the primitive variables
    P[0, :] = P0_INTERP(x_center)  # Initial propellant density ng TODO
    P[1, :] = P1_INTERP(x_center)  # Initial ni
    P[2, :] = P2_INTERP(x_center)  # Initial vi
    P[3, :] = P3_INTERP(x_center)  # Initial Te
    P[4, :] = P4_INTERP(x_center)  # Initial Ve
    P[5, :] = P5_INTERP(x_center)  # Initial Ve

    Jm1 = J_INIT
    J = J_INIT

    del (
        t_INIT,
        P_INIT,
        U_INIT,
        P_LeftGhost_INIT,
        P_RightGhost_INIT,
        J_INIT,
        V_INIT,
        B_INIT,
        x_center_INIT,
        P0_INTERP,
        P1_INTERP,
        P2_INTERP,
        P3_INTERP,
        P4_INTERP,
    )

else:
    # We initialize the primitive variables
    ng_anode = MDOT / (
        Mi * A0 * VG
    )  # Initial propellant density ng at the anode locatione = MDOT / (Mi* A0 * VG)  # Initial propellant density ng at the anode location
    # P[0,:] = InitNeutralDensity(x_center, ng_anode, VG, P, ionization_type, SIZMAX, LSIZ1, LSIZ2) # initialize n_g in the space so that it is cst in time if there is no wall recombination.
    ### Warning, in the code currently, neutrals dyanmic is canceled.
    P[0, :] = ng_anode
    P[1, :] = msp.NI0  # Initial ni
    P[2, :] = 0.0  # Initial vi
    P[3, :] = msp.TE0  # Initial Te

    def SmoothInitialTemperature(bulk_array: np.ndarray, Toutlet: float) -> np.ndarray:
        """Return a smoothed version of the array bulkarray. It contains the bulk e-
        initial temperature. It smoothes the possible jump between this bulk
        value and the cathode e- temperature. Otherwise it would introduce a
        harmful discontinuity.

        Args:
            bulk_array (np.ndarray): array containg the initial bulk T_e.
            Toutlet (float): value of T_e at the outlet. Usually the cathode T_e.
        """
        bulk_copy = np.copy(bulk_array)
        nsmooth = bulk_array.shape[0] // 10
        for i in range(nsmooth):
            a = (i + 1) / (nsmooth + 1)
            bulk_copy[-1 - i] = bulk_array[-1 - i] * a + Toutlet * (1 - a)

        return bulk_copy

    P[3, :] = SmoothInitialTemperature(P[3, :], Te_Cath)
    P[4, :] = 0.0

    Jm1 = 0.0
    J = 0.0  # Initial Current

if msp.TIMESCHEME == "TVDRK3":
    """Allocation of vectors for the TVDRK3 scheme"""
    P_1 = np.ones(
        (6, NBPOINTS)
    )  # Primitive vars P = [ng, ni, ui,  Te, ve, Ue_y] TODO: maybe add , E
    U_1 = np.ones(
        (5, NBPOINTS)
    )  # Conservative vars U = [rhog, rhoi, rhoUi, 3/2 ne*e*Te, rhoe*U_{e_y}]

if msp.Circuit:
    """Circuit parameters"""
    R = msp.R
    L = msp.L
    C = msp.C
    V0 = msp.V0
    print(f"~~~~~~~~~~~~~~~~ Circuit: R = {R:.2e} Ohm")
    print(f"~~~~~~~~~~~~~~~~ Circuit: L = {L:.2e} H")
    print(f"~~~~~~~~~~~~~~~~ Circuit: C = {C:.2e} F")

    X_Volt0 = np.zeros(2)  # [DeltaV, dDeltaV/dt]
    X_Volt1 = np.zeros(2)
    X_Volt2 = np.zeros(2)
    X_Volt3 = np.zeros(2)

    RHS_Volt0 = np.zeros(2)
    RHS_Volt1 = np.zeros(2)
    RHS_Volt2 = np.zeros(2)

    A_Volt = np.zeros([2, 2])
    A_Volt[0, 0] = 0.0
    A_Volt[0, 1] = 1.0
    A_Volt[1, 1] = -1 / (L * C)
    A_Volt[1, 0] = -1 / (R * C)

    dJdt = 0.0

i_save = 0
time = 0.0
iter = 0
J = 0.0  # Initial Current

if thomas_BM_testcase:
    """Thomas' benchmark ionization source term"""
    xm = (msp.LSIZ1 + msp.LSIZ2) / 2
    imposed_Siz = msp.SIZMAX * np.cos(
        math.pi * (x_center - xm) / (msp.LSIZ2 - msp.LSIZ1)
    )
    imposed_Siz = np.where(
        (x_center < msp.LSIZ1) | (x_center > msp.LSIZ2), 0.0, imposed_Siz
    )
    del xm
else:
    imposed_Siz = np.zeros(NBPOINTS)

###############################################
#           FUNCTIONS DEFINING OUR MODEL
###############################################


def compute_mu(fP):
    """
    Compute the effective mobility mu_eff.
    It is not used in this version of the code, but we kept the function for future use.
    """
    ng = fP[0, :]
    Te = fP[3, :]

    wce = phy_const.e * Barr / phy_const.m_e

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986

    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.
        nu_iw = (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / Mi)
        # Limit the wall interactions to the inner channel
        nu_iw[x_center > LTHR] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate
    elif wall_inter_type == "None":
        nu_iw = np.zeros(fP.shape[1], dtype=float)  # Ion - wall collision rate
        nu_ew = np.zeros(fP.shape[1], dtype=float)  # Electron - wall collision rate

    nu_m = ng * KEL + alpha_B * wce + nu_ew
    mu_eff_arr = phy_const.e / (phy_const.m_e * nu_m * (1 + (wce / nu_m) ** 2))

    return mu_eff_arr


@njit
def PrimToCons(fP, fU):
    """
    Convert primitive variables to conservative variables.
    """
    fU[0, :] = fP[0, :] * Mi  # rhog
    fU[1, :] = fP[1, :] * Mi  # rhoi
    fU[2, :] = fP[2, :] * fP[1, :] * Mi  # rhoi * Ui
    fU[3, :] = (
        0.5 * phy_const.m_e * fP[1, :] * fP[5, :] ** 2
        + 3.0 / 2.0 * fP[1, :] * phy_const.e * fP[3, :]
    )  # (1/2*rhoe*Ue_y^2 +  3/2*ni*e*Te)
    fU[4, :] = phy_const.m_e * fP[1, :] * fP[5, :]  # rhoe * Ue_y


@njit
def ConsToPrim(fU, fP, fJ=0.0):
    """
    Convert conservative variables to primitive variables.
    """
    fP[0, :] = fU[0, :] / Mi  # ng
    fP[1, :] = fU[1, :] / Mi  # ni
    fP[2, :] = fU[2, :] / fU[1, :]  # Ui = rhoUi/rhoi
    fP[3, :] = (
        2.0
        / 3.0
        * (fU[3, :] - 0.5 * fU[4, :] ** 2 / (phy_const.m_e * fU[1, :] / Mi))
        / (phy_const.e * fP[1, :])
    )  # Te
    fP[4, :] = fP[2, :] - fJ / (A0 * phy_const.e * fP[1, :])  # ve
    fP[5, :] = fU[4, :] / (phy_const.m_e * fU[1, :] / Mi)  # Ue_y


@njit
def InviscidFlux(fP, fF, tau_xy=0.0, heat_flux_vec=0.0):
    """
    Compute the inviscid flux.
    """
    fF[0, :] = fP[0, :] * VG * Mi  # rho_g*v_g
    fF[1, :] = fP[1, :] * fP[2, :] * Mi  # rho_i*v_i
    fF[2, :] = (
        Mi * fP[1, :] * fP[2, :] * fP[2, :] + fP[1, :] * phy_const.e * fP[3, :]
    )  # M*n_i*v_i**2 + p_e
    fF[3, :] = (
        (
            5.0 / 2.0 * fP[1, :] * phy_const.e * fP[3, :]
            + 0.5 * phy_const.m_e * fP[1, :] * fP[5, :] ** 2
        )
        * fP[4, :]
        + tau_xy * fP[5, :]
        + heat_flux_vec
    )  # (1/2*rhoe*uey^2*v_e + 5/2n_i*e*T_e*v_e)
    fF[4, :] = phy_const.m_e * fP[1, :] * fP[5, :] * fP[4, :]  # (rhoe * uey * uex)

@njit
def gradient(y, x):
    """
    Compute the gradient of a function y(x) using a second order centered difference scheme.
    """
    dp_dz = np.zeros(y.shape)
    dp_dz[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
    dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

    return dp_dz


@njit
def compute_E(fP):
    '''
    Compute the electric field E.
    '''
    # TODO: This is already computed! Maybe move to the source
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0, :]
    ni = fP[1, :]
    Te = fP[3, :]
    ve = fP[4, :]
    Ue_y = fP[5, :]

    me = phy_const.m_e
    wce = phy_const.e * Barr / me  # electron cyclotron frequency

    #############################
    #       Compute the rates   #
    #############################

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.
        nu_iw = (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / Mi)
        # Limit the wall interactions to the inner channel
        nu_iw[x_center > LTHR] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate

    elif wall_inter_type == "None":
        nu_iw = np.zeros(Te.shape, dtype=float)  # Ion - wall collision rate
        nu_ew = np.zeros(Te.shape, dtype=float)  # Electron - wall collision rate

    # TODO: Put decreasing wall collisions (Not needed for the moment)
    #    if decreasing_nu_iw:
    #        index_L1 = np.argmax(z > L1)
    #        index_LTHR = np.argmax(z > LTHR)
    #        index_ind = index_L1 - index_LTHR + 1
    #
    #        nu_iw[index_LTHR: index_L1] = nu_iw[index_LTHR] * np.arange(index_ind, 1, -1) / index_ind
    #        nu_iw[index_L1:] = 0.0

    ##################################################
    #       Compute the electron properties          #
    ##################################################
    nu_m = (
        ng * KEL + alpha_B * wce + nu_ew
    )  # Electron momentum - transfer collision frequency

    if sum(Ue_y) != 0.0:

        div_p = gradient(
            phy_const.e * ni * Te, x_center
        )  # To be used with 5./2 and + div_p*ve in line 231
        E = -Ue_y * Barr - div_p / (phy_const.e * ni)  # electric field
    else:
        # old scheme without Ue_y
        mu_eff = (phy_const.e / (me * nu_m)) * (
            1.0 / (1 + (wce / nu_m) ** 2)
        )  # Effective mobility    dp_dz  = np.gradient(ni*Te, Delta_x)

        dp_dz = gradient(ni * Te, x_center)
        E = -ve / mu_eff - dp_dz / ni  # Discharge electric field

    return E


@njit
def Source(fP, fS):
    '''
    Compute the source terms.
    '''
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0, :]
    ni = fP[1, :]
    vi = fP[2, :]
    Te = fP[3, :]
    ve = fP[4, :]
    Ue_y = fP[5, :]

    #############################
    #     Compute the rates     #
    #############################
    Siz_arr = np.zeros(
        ng.shape, dtype=float
    )  # the final unit of Siz_arr is m^(-3).s^(-1)
    # Computing ionization source term:
    if not boolSizImposed:
        Kiz = (
            1.8e-13 * (((1.5 * Te) / Eion) ** 0.25) * np.exp(-4 * Eion / (3 * Te))
        )  # Ion - neutral  collision rate          MARTIN: Change
        Siz_arr = ng * ni * Kiz
    else:
        Siz_arr = imposed_Siz

    # If ionization collision are considered in the momentum and energy equations.
    if boolIonColl:
        d_IC = 1.0
    else:
        d_IC = 0.0

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.
        nu_iw = (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / Mi)
        # Limit the wall interactions to the inner channel
        nu_iw[x_center > LTHR] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate
    elif wall_inter_type == "None":
        nu_iw = np.zeros(Te.shape, dtype=float)  # Ion - wall collision rate
        nu_ew = np.zeros(Te.shape, dtype=float)  # Electron - wall collision rate

    # TODO: Put decreasing wall collisions (Not needed for the moment)
    #    if decreasing_nu_iw:
    #        index_L1 = np.argmax(z > L1)
    #        index_LTHR = np.argmax(z > LTHR)
    #        index_ind = index_L1 - index_LTHR + 1
    #
    #        nu_iw[index_LTHR: index_L1] = nu_iw[index_LTHR] * np.arange(index_ind, 1, -1) / index_ind
    #        nu_iw[index_L1:] = 0.0

    ##################################################
    #       Compute the electron properties          #
    ##################################################
    phi_W = Te * np.log(np.sqrt(Mi / (2 * np.pi * me)) * (1 - sigma))  # Wall potential
    Ew = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    nu_m = (
        ng * KEL + alpha_B * wce + nu_ew
    )  # Electron momentum - transfer collision frequency

    # if the empirical term is used, we compute it here
    if np.any(empirical_term_interp_y) != 0.0:
        RieY = np.copy(empirical_term_interp_y)
        RieX = np.copy(empirical_term_interp_x)
        # RieY -= Rei_sat(ni, Te, vi, fDelta_x[0], fMi)
    else:
        RieY = -phy_const.m_e * nu_m * ni * Ue_y
        RieX = -phy_const.m_e * ni * nu_m * ve

    # mu_eff = (phy_const.e / (me* nu_m)) * (
    #     1.0 / (1 + (wce / nu_m) ** 2)
    #     )  # Effective mobility

    # div_u   = gradient(ve, d=Delta_x)               # To be used with 3./2. in line 160 and + phy_const.e*ni*Te*div_u  in line 231

    if boolPressureDiv:  # TODO: check that is is verified by default
        div_p = gradient(
            phy_const.e * ni * Te, x_center
        )  # To be used with 5./2 and + div_p*ve in line 231
    else:
        div_p = np.zeros(Te.shape)  # this line to match the old version of the code.

    E_x = -Ue_y * Barr - div_p / (phy_const.e * ni)

    # Compute the source terms
    fS[0, :] = (-d_IC * Siz_arr + nu_iw * ni) * Mi  # Gas Density
    fS[1, :] = (Siz_arr - nu_iw * ni) * Mi  # Ion Density
    fS[2, :] = (
        d_IC * Siz_arr * VG * Mi
        + RieX
        - phy_const.e * ni * Barr * Ue_y
        - nu_iw * ni * vi * Mi
    )  # Momentum electrons axial
    fS[3, :] = (
        -d_IC * Siz_arr * Eion * gamma_i * phy_const.e
        - nu_ew * ni * Ew * phy_const.e
        - phy_const.e * ni * E_x * ve
        + 1.5 * Siz_arr * phy_const.e * 10.0  #
        - 0.5 * Siz_arr * phy_const.m_e * Ue_y**2  # new term
    )  # Electron energy
    fS[4, :] = RieY + phy_const.e * ni * Barr * ve  # Momentum electrons azimuthal


@njit
def heatFlux(fP, fS):
    '''
    Compute the heat flux.
    '''
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0, :]  # It contains ghost cells
    ni = fP[1, :]  # It contains ghost cells
    Te = fP[3, :]  # It contains ghost cells

    #############################
    #       Compute the rates   #
    #############################

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.
        nu_iw = (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / Mi)
        # Limit the wall interactions to the inner channel
        nu_iw[x_center > LTHR] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate

    elif wall_inter_type == "None":
        nu_iw = np.zeros(Te.shape, dtype=float)  # Ion - wall collision rate
        nu_ew = np.zeros(Te.shape, dtype=float)  # Electron - wall collision rate

    # TODO: Put decreasing wall collisions (Not needed for the moment)
    #    if decreasing_nu_iw:
    #        index_L1 = np.argmax(z > L1)
    #        index_LTHR = np.argmax(z > LTHR)
    #        index_ind = index_L1 - index_LTHR + 1
    #
    #        nu_iw[index_LTHR: index_L1] = nu_iw[index_LTHR] * np.arange(index_ind, 1, -1) / index_ind
    #        nu_iw[index_L1:] = 0.0

    ##################################################
    #       Compute the electron properties          #
    ##################################################
    phi_W = Te * np.log(np.sqrt(Mi / (2 * np.pi * me)) * (1 - sigma))  # Wall potential
    Ew = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    nu_m = ng * KEL + alpha_B * wce + nu_ew  #

    kappa = 5.0 / 2.0 * ni * phy_const.e**2 * Te / (phy_const.m_e * nu_m)
    kappa_perp = kappa / (1 + (wce / nu_m) ** 2)

    # kappa_12 = 0.5*(kappa[1:] + kappa[:-1])
    kappa_12 = 0.5 * (kappa_perp[1:] + kappa_perp[:-1])
    grad_Te = (Te[1:] - Te[:-1]) / (x_center[1:] - x_center[:-1])

    q_12 = -0.5 * kappa_12 * grad_Te  # 1/2 test just to match P.A. data
    q_source = (q_12[1:] - q_12[:-1]) / Delta_x

    # fS[0, :] = (-Siz_arr + nu_iw[:] * ni[:]) * Mi # Gas Density

    fS[3, :] += -q_source
    delta_T_min = CFL * np.min((ni[1:-1] * phy_const.e * Delta_x**2) / kappa_perp[1:-1])

    return delta_T_min

    # + phy_const.e*ni*Te*div_u  #- gradI_term*ni*Te*grdI          # Energy in Joule


@njit
def TDMA(
    a, b, c, d
):  # Thomas algorithm for the implicit solver a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    n = len(d)
    w = np.zeros(n - 1, float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
    return p


@njit
def heatFluxImplicit(fP, fDelta_t):
    '''
    Compute the heat flux with an implicit scheme.
    '''
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0, :]  # It contains ghost cells
    ni = fP[1, :]  # It contains ghost cells
    Te = fP[3, :]  # It contains ghost cells

    me = phy_const.m_e
    wce = phy_const.e * Barr_extended / me  # electron cyclotron frequency

    #############################
    #       Compute the rates   #
    #############################

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.
        nu_iw = (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / Mi)
        # Limit the wall interactions to the inner channel
        nu_iw[x_center > LTHR] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate

    elif wall_inter_type == "None":
        nu_iw = np.zeros(Te.shape, dtype=float)  # Ion - wall collision rate
        nu_ew = np.zeros(Te.shape, dtype=float)  # Electron - wall collision rate

    # TODO: Put decreasing wall collisions (Not needed for the moment)
    #    if decreasing_nu_iw:
    #        index_L1 = np.argmax(z > L1)
    #        index_LTHR = np.argmax(z > LTHR)
    #        index_ind = index_L1 - index_LTHR + 1
    #
    #        nu_iw[index_LTHR: index_L1] = nu_iw[index_LTHR] * np.arange(index_ind, 1, -1) / index_ind
    #        nu_iw[index_L1:] = 0.0

    ##################################################
    #       Compute the electron properties          #
    ##################################################
    phi_W = Te * np.log(np.sqrt(Mi / (2 * np.pi * me)) * (1 - sigma))  # Wall potential
    Ew = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    nu_m = ng * KEL + alpha_B * wce + nu_ew  #

    kappa = 5.0 / 2.0 * ni * phy_const.e**2 * Te / (phy_const.m_e * nu_m)
    kappa_perp = kappa / (1 + (wce / nu_m) ** 2)

    # kappa_12 = 0.5*(kappa[1:] + kappa[:-1])
    kappa_12 = 0.5 * (kappa_perp[1:] + kappa_perp[:-1])

    # Simple implicit
    Coefficient = 2.0 / 3.0 * fDelta_t / (ni * phy_const.e * Delta_x)

    a_lowerDiag = -Coefficient[1:-1] * kappa_12[:-1] / (x_center[1:-1] - x_center[:-2])
    b_mainDiag = np.ones_like(Coefficient[1:-1]) + Coefficient[1:-1] * (
        kappa_12[:-1] * (x_center[2:] - x_center[1:-1])
        + kappa_12[1:] * (x_center[1:-1] - x_center[:-2])
    ) / ((x_center[1:-1] - x_center[:-2]) * (x_center[2:] - x_center[1:-1]))
    c_upperDiag = -Coefficient[1:-1] * kappa_12[1:] / (x_center[2:] - x_center[1:-1])

    d_solutionVector = np.copy(Te[1:-1])
    d_solutionVector[0] -= a_lowerDiag[0] * Te[0]  # Adding the boundary conditions
    d_solutionVector[-1] -= c_upperDiag[-1] * Te[-1]

    # # Crank Nicolson constant dx
    # Coefficient = 0.5*2. / 3. * fDelta_t/( ni * fDelta_x**2)

    # a_lowerDiag = -Coefficient[1:-1]*kappa_12[:-1]
    # b_mainDiag  = np.ones_like(Coefficient[1:-1]) + Coefficient[1:-1]*(kappa_12[:-1] + kappa_12[1:])
    # c_upperDiag = -Coefficient[1:-1]*kappa_12[1:]

    # d_solutionVector      = np.copy(Te[1:-1]) - a_lowerDiag*Te[:-2] - c_upperDiag*Te[2:] + b_mainDiag*Te[1:-1]
    # d_solutionVector[0]  += a_lowerDiag[0]*Te[0]     # Adding the boundary conditions
    # d_solutionVector[-1] += c_upperDiag[-1]*Te[-1]

    return TDMA(a_lowerDiag[1:], b_mainDiag, c_upperDiag[:-1], d_solutionVector)


@njit
def simpson(y, x):
    """
    Simpson's rule for integration
    y is the vector of values, x is the vector of corresponding x values
    """
    dx = x[1] - x[0]
    return dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])


@njit
def calculate_Rei(ne, Te, uey):
    """Calculate the theoretical electron-ion collision friction using a Maxwellian distribution"""
    lambda_D = (
        (phy_const.epsilon_0 * Te * phy_const.elementary_charge)
        / (ne * phy_const.elementary_charge**2)
    ) ** 0.5
    omega_pe = (ne * phy_const.elementary_charge) / (
        phy_const.electron_mass * phy_const.epsilon_0
    ) ** 0.5
    Ewave = 1.5 * ne * Te * phy_const.elementary_charge / 432
    vTe = (2 * Te * phy_const.elementary_charge / phy_const.electron_mass) ** 0.5
    Rei_Maxwellian = (
        4
        * (2 * np.pi) ** 0.5
        * omega_pe
        * lambda_D
        * Ewave
        * uey
        / vTe**3
        * np.exp(-((uey / vTe) ** 2))
    )
    return Rei_Maxwellian


@njit
def Rei_sat(ne, Te, vix, dx, mass):
    """
    Calculate the saturated electron-ion collision friction using the empirical formula
    """
    grad_term = np.gradient(vix * ne * Te, dx)
    cs = phy_const.elementary_charge * Te / mass
    return phy_const.elementary_charge / (16 * 6**0.5 * cs) * np.abs(grad_term)


# Compute the Current
@njit
def compute_I(fP, fV, old_curr=True, n_old_U_ey_old=0.0):
    """Compute the discharge current using the old or the new scheme"""

    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0, :]
    ni = fP[1, :]
    vi = fP[2, :]
    Te = fP[3, :]
    ve = fP[4, :]
    Ue_y = fP[5, :]

    #############################
    #       Compute the rates   #
    #############################
    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.
        nu_iw = (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / Mi)
        # Limit the collisions to inside the thruster
        index_LTHR = np.argmax(x_center > LTHR)
        nu_iw[index_LTHR:] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate
    elif wall_inter_type == "None":
        nu_iw = np.zeros(Te.shape, dtype=float)  # Ion - wall collision rate
        nu_ew = np.zeros(Te.shape, dtype=float)  # Electron - wall collision rate

    if old_curr:
        # Electron momentum - transfer collision frequency
        nu_m = ng * KEL + alpha_B * wce + nu_ew

        div_p = gradient(ni * Te, x_center)

        Term_1 = +Ue_y * Barr + phy_const.m_e / phy_const.e * nu_m * vi + div_p / (ni)
        value_simpson_1 = simpson(Term_1, x_center)

        top = fV + value_simpson_1
        Term_2 = phy_const.m_e / phy_const.e * nu_m / ni
        value_simpson_2 = simpson(Term_2, x_center)
        bottom = phy_const.e * A0 * Rext + value_simpson_2
        J0 = top / bottom  # Discharge current density
    else:
        nu_m = ng * KEL + alpha_B * wce + nu_ew

        div_p = gradient(ni * Te, x_center)
        div_mnuxuy = gradient(me * ni * ve * Ue_y, x_center)
        div_uey = gradient(Ue_y, x_center)

        if np.sum(n_old_U_ey_old) == 0.0:
            n_old_U_ey_old = np.copy(ni * Ue_y)
        dt_m_n_uey = phy_const.m_e * (ni * Ue_y - n_old_U_ey_old) / Delta_t

        if np.any(empirical_term_interp_y) != 0:
            # use the interpolated empirical term
            RieY = np.copy(empirical_term_interp_y)
            RieX = np.copy(empirical_term_interp_x)
            # RieY -= Rei_sat(ni, Te, vi, fDelta_x[0], fMi)
        else:
            RieX = -phy_const.m_e * nu_m * ni * ve
            RieY = -phy_const.m_e * nu_m * ni * Ue_y

        Term_1 = Ue_y * Barr + div_p / ni - RieX / (phy_const.e * ni) + vi * Barr
        Term_1 += (
            RieY / (phy_const.e * ni)
            - div_mnuxuy / (phy_const.e * ni)
            - dt_m_n_uey / (phy_const.e * ni)
        )
        # value_simpson_1 = integrate.simpson(Term_1 , x=fx_center)
        Term_1 = np.append(Term_1, Term_1[-1] + Term_1[-1] - Term_1[-2])
        value_simpson_1 = simpson(
            Term_1, np.append(x_center, x_center[-1] + x_center[1] - x_center[0])
        )
        top = fV + value_simpson_1

        Term_2 = Barr / ni - phy_const.electron_mass * div_uey / (phy_const.e * ni)
        # value_simpson_2 = integrate.simpson(Term_2 , x=x_center) # TODO: check if this is correct with the other simpson
        Term_2 = np.append(Term_2, Term_2[-1] + Term_2[-1] - Term_2[-2])

        value_simpson_2 = simpson(
            Term_2, np.append(x_center, x_center[-1] + x_center[1] - x_center[0])
        )
        bottom = phy_const.e * A0 * Rext + value_simpson_2
        J0 = top / bottom  # Discharge current density

    return J0 * phy_const.e * A0


@njit
def SetInlet(fP_LeftColumn, fU_ghost, fP_ghost, fJ=0.0, moment=1):
    """Impose the left boundary conditions"""
    # TODO: change the Dirichlet BCs so that a fixed value s is achieved in the frontier x=0. So the ghost value must be s_g = 2*s - s[0], where s[0] is the left value of the bulk array. Currently only v_i is computed this way to achieve the Bohm velocity at the frontier. It is not the case for n_g and T_e.
    fP_LC = fP_LeftColumn  # renaming for more elegance

    U_Bohm = np.sqrt(phy_const.e * fP_LC[3] / Mi)

    if not boolSizImposed:
        if fP_LC[1] * fP_LC[2] < 0.0:
            fU_ghost[0] = (MDOT - Mi * fP_LC[1] * fP_LC[2] * A0) / (A0 * VG)
        else:
            fU_ghost[0] = MDOT / (A0 * VG)

    else:
        fU_ghost[0] = MDOT / (A0 * VG)

    fU_ghost[1] = fP_LC[1] * Mi
    fU_ghost[2] = (
        -2.0 * fP_LC[1] * U_Bohm * Mi - fP_LC[1] * fP_LC[2] * Mi
    )  # so that 0.5*(U_ghost[2] + U_LC[2]) = - u_B * U_LC[1]
    U_ey_In = fP_LC[5]
    # Energy_Ghost =  0.5 * phy_const.m_e * fP_LC[1] * fP_LC[5]**2 + 3.0 / 2.0 * fP_LC[1] * phy_const.e * Te_Cath #(2*fTe_cath - fP_In[3])
    # fU_ghost[3] = Energy_Ghost
    fU_ghost[3] = (
        0.5 * phy_const.m_e * fP_LC[1] * fP_LC[5] ** 2
        + 3.0 / 2.0 * fP_LC[1] * phy_const.e * fP_LeftColumn[3]
    )  # (2*fTe_cath - fP_In[3])
    fU_ghost[4] = phy_const.m_e * fP_LC[1] * fP_LC[5]

    fP_ghost[0] = fU_ghost[0] / Mi  # ng
    fP_ghost[1] = fU_ghost[1] / Mi  # ni
    fP_ghost[2] = fU_ghost[2] / fU_ghost[1]  # Ui
    fP_ghost[3] = (
        2.0
        / 3.0
        * (fU_ghost[3] - 0.5 * fU_ghost[4] ** 2 / (phy_const.m_e * fU_ghost[1] / Mi))
        / (phy_const.e * fP_ghost[1])
    )  # Te
    fP_ghost[4] = fP_ghost[2] - fJ / (A0 * phy_const.e * fP_ghost[1])  # ve
    fP_ghost[5] = fU_ghost[4] / (phy_const.m_e * fU_ghost[1] / Mi)  # Ue_y


@njit
def SetOutlet(fP_RightColumn, fU_ghost, fP_ghost, J=0.0):
    """Impose the right boundary conditions"""
    # TODO: change the Dirichlet BCs so that a fixed value s is achieved in the frontier x=0. So the ghost value must be s_g = 2*s - s[0], where s[0] is the left value of the bulk array. It is not the case for T_e.
    fP_RC = fP_RightColumn  # renaming for more elegance

    fU_ghost[0] = fP_RC[0] * Mi
    fU_ghost[1] = fP_RC[1] * Mi
    fU_ghost[2] = fP_RC[1] * fP_RC[2] * Mi
    fU_ghost[3] = 3.0 / 2.0 * fP_RC[1] * phy_const.e * Te_Cath
    fU_ghost[4] = phy_const.m_e * fP_RC[1] * 0.0  # TODO: check if this is correct

    fP_ghost[0] = fU_ghost[0] / Mi  # ng
    fP_ghost[1] = fU_ghost[1] / Mi  # ni
    fP_ghost[2] = fU_ghost[2] / fU_ghost[1]  # Ui
    fP_ghost[3] = (
        2.0
        / 3.0
        * (fU_ghost[3] - 0.5 * fU_ghost[4] ** 2 / (phy_const.m_e * fU_ghost[1] / Mi))
        / (phy_const.e * fP_ghost[1])
    )  # Te
    fP_ghost[4] = fP_ghost[2] - J / (A0 * phy_const.e * fP_ghost[1])  # ve
    fP_ghost[5] = fU_ghost[4] / (phy_const.m_e * fU_ghost[1] / Mi)  # Ue_y


##########################################################
#           Functions defining our numerics              #
##########################################################
# TODO: These are vector. Better allocate them
@njit
def computeMaxEigenVal_e(fP):

    U_Bohm = np.sqrt(phy_const.e * fP[3, :] / Mi)

    return np.maximum(np.abs(U_Bohm - fP[4, :]) * 2, np.abs(U_Bohm + fP[4, :]) * 2)


@njit
def computeMaxEigenVal_i(fP):

    U_Bohm = np.sqrt(phy_const.e * fP[3, :] / Mi)

    # return [max(l1, l2) for l1, l2 in zip(abs(U_Bohm - P[2,:]), abs(U_Bohm + P[2,:]))]
    return np.maximum(np.abs(U_Bohm - fP[2, :]), np.abs(U_Bohm + fP[2, :]))


@njit
def NumericalFlux(fP, fU, fF_cell, fF_interf):

    # Compute the max eigenvalue
    lambda_max_i_R = computeMaxEigenVal_i(fP[:, 1 : NBPOINTS + 2])
    lambda_max_i_L = computeMaxEigenVal_i(fP[:, 0 : NBPOINTS + 1])
    lambda_max_i_12 = np.maximum(lambda_max_i_L, lambda_max_i_R)

    lambda_max_e_R = computeMaxEigenVal_e(fP[:, 1 : NBPOINTS + 2])
    lambda_max_e_L = computeMaxEigenVal_e(fP[:, 0 : NBPOINTS + 1])
    lambda_max_e_12 = np.maximum(lambda_max_e_L, lambda_max_e_R)

    # Compute the flux at the interface
    fF_interf[0, :] = 0.5 * (
        fF_cell[0, 0 : NBPOINTS + 1] + fF_cell[0, 1 : NBPOINTS + 2]
    ) - 0.5 * VG * (fU[0, 1 : NBPOINTS + 2] - fU[0, 0 : NBPOINTS + 1])

    fF_interf[1, :] = 0.5 * (
        fF_cell[1, 0 : NBPOINTS + 1] + fF_cell[1, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i_12 * (fU[1, 1 : NBPOINTS + 2] - fU[1, 0 : NBPOINTS + 1])

    fF_interf[2, :] = 0.5 * (
        fF_cell[2, 0 : NBPOINTS + 1] + fF_cell[2, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i_12 * (fU[2, 1 : NBPOINTS + 2] - fU[2, 0 : NBPOINTS + 1])

    fF_interf[3, :] = 0.5 * (
        fF_cell[3, 0 : NBPOINTS + 1] + fF_cell[3, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_e_12 * (fU[3, 1 : NBPOINTS + 2] - fU[3, 0 : NBPOINTS + 1])

    fF_interf[4, :] = 0.5 * (
        fF_cell[4, 0 : NBPOINTS + 1] + fF_cell[4, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_e_12 * (fU[4, 1 : NBPOINTS + 2] - fU[4, 0 : NBPOINTS + 1])


@njit
def ComputeDelta_t(fP):

    x_ext = x_center_extended  # renaming for elegance

    # Compute the max eigenvalue
    lambda_max_i_R = computeMaxEigenVal_i(fP[:, 1 : NBPOINTS + 2])
    lambda_max_i_L = computeMaxEigenVal_i(fP[:, 0 : NBPOINTS + 1])
    lambda_max_i_12 = np.maximum(lambda_max_i_L, lambda_max_i_R)

    lambda_max_e_R = computeMaxEigenVal_e(fP[:, 1 : NBPOINTS + 2])
    lambda_max_e_L = computeMaxEigenVal_e(fP[:, 0 : NBPOINTS + 1])
    lambda_max_e_12 = np.maximum(lambda_max_e_L, lambda_max_e_R)

    # Delta_t = CFL * fDelta_x / (max(max(lambda_max_e_12), max(lambda_max_i_12)))
    Delta_t = CFL * min(
        (x_ext[1:] - x_ext[:-1]) / np.maximum(lambda_max_e_12, lambda_max_i_12)
    )

    return Delta_t


########################################################################################################################
############### START THE MAIN LOOP ####################################################################################
########################################################################################################################


# We initialize the conservative variables
PrimToCons(P, U)

if TIMESCHEME == "Forward Euler":
    ##########################################################################################
    #           Loop with Forward Euler                                                      #
    #           U^{n+1}_j = U^{n}_j - Dt/Dx(F^n_{j+1/2} - F^n_{j-1/2}) + Dt S^n_j            #
    ##########################################################################################
    print("Using Forward Euler scheme")
    n_old_U_ey_old = np.copy(P[1, :] * P[5, :])
    J = compute_I(P, V, False, n_old_U_ey_old)

    while time < TIMEFINAL:

        # Save results
        if (iter % SAVERATE) == 0:
            # Compute the electric field.
            Efield = compute_E(P)
            # Save the variant data
            filenameTemp = ResultsData + "/MacroscopicVars_" + f"{i_save:06d}" + ".pkl"
            pickle.dump(
                [time, P, U, P_LeftGhost, P_RightGhost, J, Efield, V],
                open(filenameTemp, "wb"),
            )
            i_save += 1
            print(
                "Iter = ",
                iter,
                "\tTime = {:.2f} µs".format(time / 1e-6),
                "\tI = {:.4f} A".format(J),
                "\tJ = {:.3e} A/m2".format(J / A0),
            )

        # Set the boundaries
        SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, J, 1)
        SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(
            np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
            F_cell,
            tau_xy,
            heat_flux,
        )

        # Compute the convective Delta t
        Delta_t = ComputeDelta_t(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1))

        # print("Delta_t = ", Delta_t)

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
            np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
            F_cell,
            F_interf,
        )

        # Compute the source in the center of the cell
        Source(P, S)

        if HEATFLUX and not IMPlICIT:
            dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S)
            Delta_t = min(dt_HF, Delta_t)
        elif HEATFLUX and IMPlICIT:
            dt_HF = Delta_t
            Te = heatFluxImplicit(
                np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1)
            )

        # Update the solution
        U[:, :] = (
            U[:, :]
            - Delta_t
            / Delta_x
            * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
            + Delta_t * S[:, :]
        )

        # Prevent the energy to be strictly negative
        # U[3,:] = np.where(U[3,:] >= 0., U[3,:], 0.)

        # Compute the current
        J = compute_I(P, V, False, n_old_U_ey_old)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, J)

        n_old_U_ey_old = np.copy(P[1, :] * P[5, :])

        P[3, :] = np.where(P[3, :] >= T_min, P[3, :], T_min)
        P_LeftGhost[3] = T_min if P_LeftGhost[3] <= T_min else P_LeftGhost[3]
        P_RightGhost[3] = T_min if P_RightGhost[3] <= T_min else P_RightGhost[3]
        PrimToCons(P, U)

        # Update the time
        time += Delta_t
        iter += 1

if TIMESCHEME == "TVDRK3":
    print("Using TVDRK3 scheme")

    while time < TIMEFINAL:

        # Save results
        if (iter % SAVERATE) == 0:
            # Compute the electric field.
            Efield = compute_E(P)
            filenameTemp = ResultsData + "/MacroscopicVars_" + f"{i_save:06d}" + ".pkl"
            pickle.dump(
                [time, P, U, P_LeftGhost, P_RightGhost, J, Efield, V],
                open(filenameTemp, "wb"),
            )
            i_save += 1
            print(
                "Iter = ",
                iter,
                "\tTime = {:.2f}~µs".format(time / 1e-6),
                "\tI = {:.4f}~A".format(J),
                "\tJ = {:.3e} A/m2".format(J / A0),
            )

        #################################################
        #           FIRST STEP RK3
        #################################################

        # Copy the solution to store it
        U_1[:, :] = U[:, :]
        ConsToPrim(U_1, P_1, J)
        J_1 = compute_I(P, V)
        ConsToPrim(U_1, P_1, J_1)

        # Set the boundaries
        SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, J)
        SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, J)
        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell)
        # Compute the convective Delta t (Only in the first step)
        Delta_t = ComputeDelta_t(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1))

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
            np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
            F_cell,
            F_interf,
        )

        # Compute the source in the center of the cell
        Source(P, S)

        if HEATFLUX and not IMPlICIT:
            dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S)
            Delta_t = min(dt_HF, Delta_t)
        # First half step of strang-splitting
        elif HEATFLUX and IMPlICIT:
            dt_HF = Delta_t
            P[3, :] = heatFluxImplicit(
                np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1)
            )
            U[3, :] = 3.0 / 2.0 * P[1, :] * phy_const.e * P[3, :]

        # Update the solution
        U[:, :] = (
            U[:, :]
            - Delta_t
            / Delta_x
            * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
            + Delta_t * S[:, :]
        )

        # Compute the current
        J = compute_I(P, V)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, J)

        # Compute RLC boolCircuit
        if boolCircuit:
            dJdt = (J - Jm1) / Delta_t

            RHS_Volt0[0] = X_Volt0[1]
            RHS_Volt0[1] = (
                -1 / (R * C) * X_Volt0[1] - 1.0 / (L * C) * X_Volt0[0] + 1 / C * dJdt
            )
            X_Volt1 = X_Volt0 + Delta_t * RHS_Volt0

        #################################################
        #           SECOND STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, J, 2)
        SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell)

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
            np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
            F_cell,
            F_interf,
        )

        # Compute the source in the center of the cell
        Source(
            P,
            S,
        )
        if HEATFLUX and not IMPlICIT:
            dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S)

        # Update the solution
        U[:, :] = (
            0.75 * U_1[:, :]
            + 0.25 * U[:, :]
            + 0.25
            * (
                -Delta_t
                / Delta_x
                * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
                + Delta_t * S[:, :]
            )
        )

        # Compute the current
        J = compute_I(P, V)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, J)

        # Compute RLC Circuit
        if boolCircuit:
            dJdt = (J - Jm1) / Delta_t
            RHS_Volt1[0] = X_Volt1[1]
            RHS_Volt1[1] = (
                -1 / (R * C) * X_Volt1[1] - 1.0 / (L * C) * X_Volt1[0] + 1 / C * dJdt
            )
            X_Volt2 = 0.75 * X_Volt0 + 0.25 * X_Volt1 + 0.25 * Delta_t * RHS_Volt1

        #################################################
        #           THIRD STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, J, 3)
        SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell)

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
            np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
            F_cell,
            F_interf,
        )
        # Compute the source in the center of the cell
        Source(P, S)
        if HEATFLUX and not IMPlICIT:
            dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S)

        # Update the solution
        U[:, :] = (
            1.0 / 3.0 * U_1[:, :]
            + 2.0 / 3.0 * U[:, :]
            + 2.0
            / 3.0
            * (
                -Delta_t
                / Delta_x
                * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
                + Delta_t * S[:, :]
            )
        )
        # Second half step of strang-splitting
        if HEATFLUX and IMPlICIT:
            SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, J, 3)
            SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, J)
            P[3, :] = heatFluxImplicit(
                np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1)
            )
            U[3, :] = 3.0 / 2.0 * P[1, :] * phy_const.e * P[3, :]

        # Compute the current
        J = compute_I(P, V)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, J)

        # Compute RLC Circuit
        if boolCircuit:
            dJdt = (J - Jm1) / Delta_t
            RHS_Volt2[0] = X_Volt2[1]
            RHS_Volt2[1] = (
                -1 / (R * C) * X_Volt2[1] - 1.0 / (L * C) * X_Volt2[0] + 1 / C * dJdt
            )
            X_Volt3 = (
                1.0 / 3.0 * X_Volt0
                + 2.0 / 3.0 * X_Volt2
                + 2.0 / 3.0 * Delta_t * RHS_Volt2
            )

            # Reinitialize for the Circuit
            Jm1 = J
            X_Volt0[:] = X_Volt3[:]

            # Change the Voltage
            V = V0 - X_Volt0[0]
        time += Delta_t
        iter += 1

# Saves the last frame
Efield = compute_E(P)
filenameTemp = ResultsData + "/MacroscopicVars_" + f"{i_save:06d}" + ".pkl"
pickle.dump(
    [time, P, U, P_LeftGhost, P_RightGhost, J, Efield, V], open(filenameTemp, "wb")
)
i_save += 1
print(
    "Iter = ",
    iter,
    "\tTime = {:.2f}~µs".format(time / 1e-6),
    "\tI = {:.4f}~A".format(J),
    "\tJ = {:.3e} A/m2".format(J / A0),
)

ttime_end = ttime.time()
print("Exec time = {:.2f} s".format(ttime_end - tttime_start))
