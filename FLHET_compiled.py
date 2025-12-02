import numpy as np
import scipy.constants as phy_const
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import configparser
import sys
from numba import njit
import time as ttime

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
# P = [ng, ni, ui,  Te, ve]              TODO: maybe add , E
#
# At the boundaries we impose
# Inlet:
#       ng = mdot/(M*A0*VG)*M
#       ui = -u_bohm
# Outlet:
#       Te = Te_Cath
#
# The user can change the PHYSICAL PARAMETERS
# or the NUMERICAL PARAMETERS
#
##########################################################

##########################################################
#           CONFIGURE PHYSICAL PARAMETERS
##########################################################

tttime_start = ttime.time()

configFile = sys.argv[1]
config = configparser.ConfigParser()
config.read(configFile)

physicalParameters = config["Physical Parameters"]

VG       = float(physicalParameters["Gas velocity"])              # Gas velocity
M        = float(physicalParameters["Ion Mass"]) * phy_const.m_u  # Ion Mass
m        = phy_const.m_e                                          # Electron mass
R1       = float(physicalParameters["Inner radius"])              # Inner radius of the thruster
R2       = float(physicalParameters["Outer radius"])              # Outer radius of the thruster
A0       = np.pi * (R2**2 - R1**2)                                # Area of the thruster
LENGTH   = float(physicalParameters["Length of axis"])            # length of Axis of the simulation
L0       = float(physicalParameters["Length of thruster"])            # length of thruster (position of B_max)
alpha_B1 = float(physicalParameters["Anomalous transport alpha_B1"])  # Anomalous transport
alpha_B2 = float(physicalParameters["Anomalous transport alpha_B2"])  # Anomalous transport
mdot     = float(physicalParameters["Mass flow"])                     # Mass flow rate of propellant
Te_Cath  = float(physicalParameters["Temperature Cathode"])           # Electron temperature at the cathode
NI0      = float(physicalParameters["Initial plasma density"])
XI1      = float(physicalParameters["Initial fraction singly"])
XI2_02   = float(physicalParameters["Initial fraction doubly 02"])        
XI2_12   = float(physicalParameters["Initial fraction doubly 12"])        
TE0      = float(physicalParameters["Initial Temperature"])  
Rext     = float(physicalParameters["Ballast resistor"])              # Resistor of the ballast
V        = float(physicalParameters["Voltage"])                       # Potential difference
Circuit  = bool(config.getboolean("Physical Parameters", "Circuit", fallback=False))  # RLC Circuit
Estar    = float(physicalParameters["Crossover energy"])  # Crossover energy

# Magnetic field configuration
MagneticFieldConfig = config["Magnetic field configuration"]

if MagneticFieldConfig["Type"] == "Default":
    print(MagneticFieldConfig["Type"] + " Magnetic Field")

    Bmax       = float(MagneticFieldConfig["Max B-field"])       # Max Mag field
    LB1        = float(MagneticFieldConfig["Length B-field 1"])  # Length for magnetic field
    LB2        = float(MagneticFieldConfig["Length B-field 2"])  # Length for magnetic field
    saveBField = bool(config.getboolean("Magnetic field configuration", "Save B-field", fallback=False))

elif MagneticFieldConfig["Type"] == "StationaryCodeBField":
    print(MagneticFieldConfig["Type"] + " Magnetic Field")

    Bmax       = float(MagneticFieldConfig["Max B-field"])  # Max Mag field
    CmagIn     = float(MagneticFieldConfig["Cmag In"])      # Length for magnetic field
    CmagOut    = float(MagneticFieldConfig["Cmag Out"])     # Length for magnetic field
    saveBField = bool(config.getboolean("Magnetic field configuration", "Save B-field", fallback=False))

##########################################################
#           NUMERICAL PARAMETERS
##########################################################
NumericsConfig = config["Numerical Parameteres"]

NBPOINTS   = int(NumericsConfig["Number of points"])    # Number of cells
SAVERATE   = int(NumericsConfig["Save rate"])           # Rate at which we store the data
CFL        = float(NumericsConfig["CFL"])               # Nondimensional size of the time step
TIMEFINAL  = float(NumericsConfig["Final time"])        # Last time of simulation
Results    = NumericsConfig["Result dir"]               # Name of result directory
TIMESCHEME = NumericsConfig["Time integration"]         # Time integration scheme

if not os.path.exists(Results):
    os.makedirs(Results, exist_ok=True)
with open(Results + "/Configuration.cfg", "w") as configfile:
    config.write(configfile)

##########################################################
#           Allocation of large vectors                  #
##########################################################

Delta_t = 1.0                                                                           # Initialization of Delta_t (do not change)
Delta_x = LENGTH / NBPOINTS

x_mesh = np.linspace(0, LENGTH, NBPOINTS + 1)                                           # Mesh in the interface
x_center = np.linspace(Delta_x, LENGTH - Delta_x, NBPOINTS)                             # Mesh in the center of cell
if MagneticFieldConfig["Type"]   == "Default":
    B0 = Bmax * np.exp(-(((x_center - L0) / LB1) ** 2.0))                               # Magnetic field within the thruster
    B0 = np.where(x_center < L0, B0, Bmax * np.exp(-(((x_center - L0) / LB2) ** 2.0)))  # Magnetic field outside the thruster
elif MagneticFieldConfig["Type"] == "StationaryCodeBField":
    B0 = Bmax * np.exp(- CmagIn*(((x_center - L0) / L0) ** 2.0))                        # Magnetic field within the thruster
    B0 = np.where(x_center < L0, B0, Bmax * np.exp(- CmagOut*(((x_center - L0) / L0) ** 2.0)))  # Magnetic field outside the thruster
alpha_B = (np.ones(NBPOINTS) * alpha_B1)                                                # Anomalous transport coefficient inside the thruster
alpha_B = np.where(x_center < L0, alpha_B, alpha_B2)                                    # Anomalous transport coefficient in the plume
alpha_B_smooth = np.copy(alpha_B)

# smooth between alpha_B1 and alpha_B2
for index in range(10, NBPOINTS - 9):
    alpha_B_smooth[index] = np.mean(alpha_B[index-10:index+10])
alpha_B = alpha_B_smooth

# Allocation of vectors
P = np.ones((9, NBPOINTS))              # Primitive vars P = [ng, n1, n02, n12, ui, v02, v12, Te, ve] TODO: maybe add , E
U = np.ones((8, NBPOINTS))              # Conservative vars U = [rhog, rho1, rho02, rho12, rhoU1, rhoU02, rhoU12, 3/2 ne*e*Te]
S = np.ones((8, NBPOINTS))              # Source Term
F_cell = np.ones((8, NBPOINTS + 2))     # Flux at the cell center. We include the Flux of the Ghost cells
F_interf = np.ones((8, NBPOINTS + 1))   # Flux at the interface
U_Inlet = np.ones((8, 1))               # Ghost cell on the left
P_Inlet = np.ones((9, 1))               # Ghost cell on the left
U_Outlet = np.ones((8, 1))              # Ghost cell on the right
P_Outlet = np.ones((9, 1))              # Ghost cell on the right
if TIMESCHEME == "TVDRK3":
    P_1 = np.ones((9, NBPOINTS))        # Primitive vars P = [ng, n1, n02, n12, ui, v02, v12, Te, ve] TODO: maybe add , E
    U_1 = np.ones((8, NBPOINTS))        # Conservative vars U = [rhog, rho1, rho02, rho12, rhoU1, rhoU02, rhoU12, 3/2 ne*e*Te]
if Circuit:
    R = float(physicalParameters["R"])
    L = float(physicalParameters["L"])
    C = float(physicalParameters["C"])
    V0 = V
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
    J0 = 0.0

anode_potential = True


if saveBField:
    plt.plot(x_center*100,B0*1e4)
    plt.plot([L0*100, L0*100], [0.,1.2*max(B0)*1e4],'k--')
    plt.xlabel("x [cm]")
    plt.ylabel("B [G]")
    plt.ylim([0,max(B0)*1e4+10])
    plt.grid()
    plt.savefig(Results+"/BfieldSTP-100_Stationary.pdf")
    plt.close()



##########################################################
#           Formulas defining our model                  #
##########################################################


@njit
def PrimToCons(P, U):
    U[0, :] = P[0, :] * M  # rhog
    U[1, :] = P[1, :] * M  # rho1
    U[2, :] = P[2, :] * M  # rho02
    U[3, :] = P[3, :] * M  # rho12
    U[4, :] = P[4, :] * P[1, :] * M  # rhoU1
    U[5, :] = P[5, :] * P[2, :] * M  # rhoU02
    U[6, :] = P[6, :] * P[3, :] * M  # rhoU12
    U[7, :] = 3.0 / 2.0 * (P[1, :]+2*(P[2, :]+P[3, :])) * phy_const.e * P[7, :]  # 3/2*ni*e*Te


@njit
def ConsToPrim(U, P, J=0.0):
    P[0, :] = U[0, :] / M  # ng
    P[1, :] = U[1, :] / M  # n1
    P[2, :] = U[2, :] / M  # n02
    P[3, :] = U[3, :] / M  # n12
    P[4, :] = U[4, :] / U[1, :]  # U1 = rhoU1/rho1
    P[5, :] = U[5, :] / U[2, :]  # U02 = rhoU02/rho02
    P[6, :] = U[6, :] / U[3, :]  # U12 = rhoU12/rho12

    P[7, :] = 2.0 / 3.0 * U[7, :] / (phy_const.e * (P[1, :] + 2*(P[2, :] + P[3, :])))  # Te
    P[8, :] = ((P[4, :]*P[1, :]+2*(P[5, :]*P[2, :]+P[6, :]*P[3, :])) - J / (A0 * phy_const.e)) / (P[1, :] + 2*(P[2, :] + P[3, :]))  # ve

@njit
def InviscidFlux(P, F):
    F[0, :] = P[0, :] * VG * M  # rho_g*v_g
    F[1, :] = P[1, :] * P[4, :] * M  # rho_i*v_i
    F[2, :] = P[2, :] * P[5, :] * M  # rho_02*v_02
    F[3, :] = P[3, :] * P[6, :] * M  # rho_12*v_12
    F[4, :] = (
        M * P[1, :] * P[4, :] * P[4, :] + P[1,:] * phy_const.e * P[7, :]
    )  # M*n_1*v_1**2 + 1*n1/(n1+2*(n02+n12))*p_e
    F[5, :] = (
        M * P[2, :] * P[5, :] * P[5, :] + 2*P[2,:] * phy_const.e * P[7, :]
    )  # M*n_02*v_02**2 + 2*n02/(n1+2*(n02+n12))*p_e
    F[6, :] = (
        M * P[3, :] * P[6, :] * P[6, :] + 2*P[3,:] * phy_const.e * P[7, :]
    )  # M*n_12*v_12**2 + 2*n12/(n1+2*(n02+n12))*p_e
    F[7, :] = 5.0 / 2.0 * (P[1, :]+2*(P[2, :]+P[3,:])) * phy_const.e * P[7, :] * P[8, :]  # 5/2n_e*e*T_e*v_e

@njit
def gradient(y, d):
    dp_dz = np.empty_like(y)
    dp_dz[1:-1] = (y[2:] - y[:-2]) / (2 * d)
    dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
    dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

    return dp_dz

@njit
def Source(P, S):

    # def compute_Kel(Te):
    #     # Xenon
    #     a =  6.25621116e-14
    #     b = -4.30874715e+00
    #     c = -1.45152836e+01
    #     d = -1.49229954e+01
    #     e = -5.68651763e+00
    #     f = 3.36357165e-01

    #     t = (1/Te)

    #     return 16./3.*a*t**f*np.exp(-b*t+ c*t**2 - d*t**3. + e*t**4)
    
    def compute_Kel(Te):
        """This function calculates the elastic scattering rate"""
        # Polynomial coefficients
        c0 = -3.04474930e+01
        c1 = 1.89683694e+00
        c2 = -6.63807968e-01
        c3 = 9.37924042e-03
        c4 = 2.19404998e-02
        c5 = -2.27126387e-03

        # Compute the natural logarithm of Te
        log_Te = np.log(Te)

        # Manually evaluate the polynomial using Horner's method (unrolled loop)
        result = c5
        result = result * log_Te + c4
        result = result * log_Te + c3
        result = result * log_Te + c2
        result = result * log_Te + c1
        result = result * log_Te + c0

        # Return the exponential of the polynomial result
        return np.exp(result)
    
    def compute_K01(Te):
        # Xenon single neutral ionization
        K0      = 3.713250982402397e-14
        epsilon = -0.8074961922061281
        A       = -0.0017611280968783634
        B       = 0.017611683126543775
        C       = 0.963776819737781

        return K0*np.exp(-epsilon/Te)*(np.log(1 + A*Te + B*Te**2))**C
    
    def compute_K02(Te):
        # Xenon double neutral ionization
        K0      = 2.9877259773209294e-15
        epsilon = 2.279306890331026
        A       = -0.0010349058788903616
        B       = 0.010351220959387678
        C       = 0.9986740377564686

        return K0*np.exp(-epsilon/Te)*(np.log(1 + A*Te + B*Te**2))**C

    def compute_K12(Te):
        # Xenon single singly charged ion ionization
        K0      = 9.071260705344802e-14
        epsilon = 25.626785037399753
        A       = -0.000026061685061096907
        B       = 0.0002606327354438319
        C       = -0.05673357419589643

        return K0*np.exp(-epsilon/Te)*(np.log(1 + A*Te + B*Te**2))**C
    
    def computeEpsilonLoss_neutrals(Te):
        def computeKprocess(Te, K0, epsilon, A, B, C):
            arg = 1 + A * Te + B * Te**2
            arg = np.maximum(arg, 1.0)  # Ensure all values are >= 1
            # if np.any(arg <= 1):
            #     arg = 1.0
            return K0 * np.exp(-epsilon / Te) * (np.log(arg)) ** C
        
        K_01 = computeKprocess(Te, 3.713250982402397e-14, 12.13, -0.0017611280968783634, 0.017611683126543775, 0.963776819737781)
        K_02 = computeKprocess(Te, 2.9877259773209294e-15, 33.1, -0.0010349058788903616, 0.010351220959387678, 0.9986740377564686)
        K_ex1 = computeKprocess(Te, 2.37016128e-14, 8.315, 7.99682247e-02, -5.91358673e-04, 4.51997276e-01)
        K_ex2 = computeKprocess(Te, 9.02951389e-15, 9.447, 3.12421531e+00, -3.01100074e-02, 5.59327899e-01)
        K_ex3 = computeKprocess(Te, 1.66394517e-14, 9.917, 2.83412200e+00, -2.66987222e-02, 6.98378384e-01)
        K_ex4 = computeKprocess(Te, 7.64651071e-15, 11.70, 7.35828827e-01, -5.08912904e-03, 1.39724961e+00)

        return (K_01*12.13 + K_02*33.1 + K_ex1*8.315 + K_ex2*9.447 + K_ex3*9.917 + K_ex4*11.70 + 3*m/M*compute_Kel(Te)*Te) / (K_01+K_02)



    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = P[0, :]
    n1 = P[1, :]
    n02 = P[2, :]
    n12 = P[3, :]
    ne = n1 + 2*(n02 + n12)
    u1 = P[4, :]
    u02 = P[5, :]
    u12 = P[6, :]
    Te = P[7, :]
    ve = P[8, :]

    energy = 3.0 / 2.0 * (n1+2*(n02+n12)) * phy_const.e * Te  # Electron internal energy
    # Gamma_E = 3./2.*ni*phy_const.e*Te*ve    # Flux of internal energy
    wce = phy_const.e * B0 / m  # electron cyclotron frequency

    #############################
    #       Compute the rates   #
    #############################
    Kel = compute_Kel(Te)  # Electron - neutral  collision rate     
    K01 = compute_K01(Te)
    K02 = compute_K02(Te)
    K12 = compute_K12(Te)
    epsilonLoss_ng = computeEpsilonLoss_neutrals(Te)


    ############################
    #       Wall Collisions    # 
    ############################     
    sigma      = 0.207*Te**(0.549)
    sigma_scl  = 1. - 8.3*np.sqrt(m/M)
    sigma[sigma > sigma_scl] = sigma_scl
    h_R = 0.3
    nu_iw      = 2 * h_R * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / M)
    index_L0 = np.argmax(x_center > L0)
    nu_iw[index_L0:] = 0.0
    nu_ew      =  nu_iw / (1 - sigma)                                        # Electron - wall collision rate

    # Eion = 12.1  # Ionization energy
    # gamma_i = 3  # Excitation coefficient
    # # Estar   = 50    # Crossover energy

    # Kiz = (
    #     1.8e-13 * (((1.5 * Te) / Eion) ** 0.25) * np.exp(-4 * Eion / (3 * Te))
    # )  # Ion - neutral  collision rate          TODO: Replace by better
    # Kel = 2.5e-13  # Electron - neutral  collision rate     TODO: Replace by good one
    # sigma = 2.0 * Te / Estar  # SEE yield
    # sigma[sigma > 0.986] = 0.986
    # nu_iw = (
    #     (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / M)
    # )  # Ion - wall collision rate
    # # Limit the collisions to inside the thruster
    # index_L0 = np.argmax(x_center > L0)
    # nu_iw[index_L0:] = 0.0

    # nu_ew = nu_iw / (1 - sigma)  # Electron - wall collision rate

    # TODO: Put decreasing wall collisions (Not needed for the moment)
    #    if decreasing_nu_iw:
    #        index_L1 = np.argmax(z > L1)
    #        index_L0 = np.argmax(z > L0)
    #        index_ind = index_L1 - index_L0 + 1
    #
    #        nu_iw[index_L0: index_L1] = nu_iw[index_L0] * np.arange(index_ind, 1, -1) / index_ind
    #        nu_iw[index_L1:] = 0.0

    ##################################################
    #       Compute the electron properties          #
    ##################################################
    phi_W  = Te * np.log(np.sqrt(M / (2 * np.pi * m)) * (1 - sigma))  # Wall potential
    Ew     = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    nu_m   = ( ng * Kel + alpha_B * wce + nu_ew)  # Electron momentum - transfer collision frequency
    mu_eff = (phy_const.e / (m * nu_m)) * (1.0 / (1 + (wce / nu_m) ** 2))  # Effective mobility

    div_p = gradient(
        phy_const.e * ((n1+2*(n02+n12)) * Te), d=Delta_x
    )  # To be used with 5./2 and + div_p*ve below

    div_xi1 = gradient(n1/ne, d=Delta_x)
    div_xi02 = gradient(2*n02/ne, d=Delta_x)
    div_xi12 = gradient(2*n12/ne, d=Delta_x)
        
    ############################

    # Continuity
    S[0, :] = (-ng[:] * ne[:] * K01[:] - ng[:] * ne[:] * K02[:] + nu_iw[:] * ne[:]) * M  # Gas Density
    S[1, :] = (ng[:] * ne[:] * K01[:] - n1[:] * ne[:] * K12[:] - nu_iw[:] * n1[:]) * M  # Singly Ion Density
    S[2, :] = (ng[:] * ne[:] * K02[:]) * M # Doubly Ion Density
    S[3, :] = (n1[:] * ne[:] * K12[:]) * M # Doubly Ion Density

    # Momentum
    S[4, :] = (
          ng[:] * ne[:] * K01[:] * VG
        - n1[:] * ne[:] * K12[:] * u1[:]
        - (phy_const.e / (mu_eff[:] * M)) * n1[:] * ve[:]
        - nu_iw[:] * n1[:] * u1[:]
        + div_xi1 * phy_const.e * ne[:] * Te[:]
    ) * M  # Singly Ion Momentum
    S[5, :] = (
          ng[:] * ne[:] * K02[:] * VG
        - (phy_const.e / (mu_eff[:] * M)) * 2*n02[:] * ve[:]
        + div_xi02 * phy_const.e * ne[:] * Te[:]
    ) * M  # Doubly Ion Momentum
    S[6, :] = (
          n1[:] * ne[:] * K12[:] * u1[:]
        - (phy_const.e / (mu_eff[:] * M)) * 2*n12[:] * ve[:]
        + div_xi12 * phy_const.e * ne[:] * Te[:]
    ) * M  # Doubly Ion Momentum

    # Energy
    S[7, :] = (
        - ng[:] * ne[:] * (K01[:]+K02[:]) * epsilonLoss_ng[:] * phy_const.e
        - n1[:] * ne[:] * K12[:] * 20.975 * phy_const.e
        - nu_ew[:] * ne[:] * Ew * phy_const.e
        + ne[:] / mu_eff[:] * (ve[:]) ** 2.0 * phy_const.e
        + div_p * ve
    )

# Compute the Current
# @njit
def compute_I(P, V):

    def compute_Kel(Te):
        """This function calculates the ionization rate"""
        # Polynomial coefficients
        c0 = -3.04474930e+01
        c1 = 1.89683694e+00
        c2 = -6.63807968e-01
        c3 = 9.37924042e-03
        c4 = 2.19404998e-02
        c5 = -2.27126387e-03

        # Compute the natural logarithm of Te
        log_Te = np.log(Te)

        # Manually evaluate the polynomial using Horner's method (unrolled loop)
        result = c5
        result = result * log_Te + c4
        result = result * log_Te + c3
        result = result * log_Te + c2
        result = result * log_Te + c1
        result = result * log_Te + c0

        # Return the exponential of the polynomial result
        return np.exp(result)

    # def trapz(y, d):
    #     return np.sum( (y[1:] + y[:-1]) )*d/2.0
    # def gradient(y, d):
    #     dp_dz = np.empty_like(y)
    #     dp_dz[1:-1] = (y[2:] - y[:-2]) / (2 * d)
    #     dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
    #     dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

    #     return dp_dz

    # TODO: This is already computed! Maybe move to the source
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = P[0, :]
    n1 = P[1, :]
    n02 = P[2, :]
    n12 = P[3, :]
    ne = n1 + 2*(n02 + n12)
    u1 = P[4, :]
    u02 = P[5, :]
    u12 = P[6, :]
    Te = P[7, :]
    ve = P[8, :]
    Gamma_1 = n1 * u1
    Gamma_02 = n02 * u02
    Gamma_12 = n12 * u12
    wce = phy_const.e * B0 / m  # electron cyclotron frequency

    #############################
    #       Compute the rates   #
    #############################
    Kel = compute_Kel(Te)  # Electron - neutral  collision rate     TODO: Replace by good one

    ############################
    #       Wall Collisions    # 
    ############################     
    sigma      = 0.207*Te**(0.549)
    sigma_scl  = 1. - 8.3*np.sqrt(m/M)
    sigma[sigma > sigma_scl] = sigma_scl
    h_R = 0.3
    nu_iw      = 2 * h_R * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / M)
    # nu_iw      = 2 * 0.5 * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / M)
    index_L0 = np.argmax(x_center > L0)
    nu_iw[index_L0:] = 0.0
    nu_ew      =  nu_iw / (1 - sigma)                                        # Electron - wall collision rate

    # TODO: OLD
    # sigma_old = 2.0 * Te / Estar  # SEE yield
    # sigma_old[sigma_old > 0.986] = 0.986
    # nu_iw = (
    #     (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / M)
    # )  # Ion - wall collision rate d
    # # Limit the collisions to inside the thruster
    # index_L0 = np.argmax(x_center > L0)
    # nu_iw[index_L0:] = 0.0

   

    nu_m = (
        ng * Kel + alpha_B * wce + nu_ew
    )  # Electron momentum - transfer collision frequency

    mu_eff = (phy_const.e / (m * nu_m)) * (
        1.0 / (1 + (wce / nu_m) ** 2)
    )  # Effective mobility

    dp_dz = np.empty_like(ne * Te)

    dp_dz[1:-1] = ((ne * Te)[2:] - (ne * Te)[:-2]) / (2 * Delta_x)
    dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
    dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

    value_trapz_1 = (
        np.sum(
            (
                (((Gamma_1 + 2*(Gamma_02+Gamma_12)) / (mu_eff * ne)) + dp_dz / ne)[1:]
                + (((Gamma_1 + 2*(Gamma_02+Gamma_12)) / (mu_eff * ne)) + dp_dz / ne)[:-1]
            )
        )
        * Delta_x
        / 2.0
    )

    if anode_potential:
        Te_anode = P[7, 0]  # Get the Te at the anode\
        # print("Te_anode = {} eV".format(Te_anode))
        Ce = (8 * phy_const.e * Te_anode / (np.pi * phy_const.m_e)) ** 0.5  # Electron thermal speed
        # print("Ce = {:.2f} m/s".format(Ce))
        Uze = P[8, 0]  # Get the Ue at the anode
        # print("Uze = {:.2f} m/s".format(Uze))
        try:
            if (Uze == 0.0) or (Ce == 0.0):
                phi_anode = 0.0
            else:
                phi_anode = Te_anode * np.log(- Ce / (4 * Uze))
        except:
            print("Error in computing phi_anode: Ce = {}, Uze = {}".format(Ce, Uze))
        # print("phi_anode = {:.2f} V".format(phi_anode))
        V_a = V - phi_anode  # Adjust the voltage by the anode potential
    else:
        V_a = V
    top = V_a + value_trapz_1

    value_trapz_2 = (
        np.sum(((1.0 / (mu_eff * ne))[1:] + (1.0 / (mu_eff * ne))[:-1])) * Delta_x / 2.0
    )
    bottom = phy_const.e * A0 * Rext + value_trapz_2

    I0 = top / bottom  # Discharge current density
    return I0 * phy_const.e * A0


@njit
def SetInlet(P_In, U_ghost, P_ghost, J=0.0, moment=1):

    U_Bohm1 = np.sqrt(5 * phy_const.e * P_In[7] / (3*M))
    U_Bohm2 = np.sqrt(2)*np.sqrt(5 * phy_const.e * P_In[7] / (3*M))

    U_ghost[0] = mdot / (A0 * VG)
    if P_In[1] * P_In[4] < 0.0:
        U_ghost[0] = U_ghost[0] - (M * P_In[1] * P_In[4] * A0) / (A0 * VG)

    if P_In[2] * P_In[5] < 0.0:
        U_ghost[0] = U_ghost[0] - (M * P_In[2] * P_In[5] * A0) / (A0 * VG)

    if P_In[3] * P_In[6] < 0.0:
        U_ghost[0] = U_ghost[0] - (M * P_In[3] * P_In[6] * A0) / (A0 * VG)

    U_ghost[1] = P_In[1] * M
    U_ghost[2] = P_In[2] * M
    U_ghost[3] = P_In[3] * M

    U_ghost[4] = -2.0 * P_In[1] * U_Bohm1 * M - P_In[1] * P_In[4] * M
    U_ghost[5] = -2.0 * P_In[2] * U_Bohm2 * M - P_In[2] * P_In[5] * M
    U_ghost[6] = -2.0 * P_In[3] * U_Bohm2 * M - P_In[3] * P_In[6] * M

    U_ghost[7] = 3.0 / 2.0 * (P_In[1]+2*(P_In[2]+P_In[3])) * phy_const.e * P_In[7]

    P_ghost[0] = U_ghost[0] / M  # ng
    P_ghost[1] = U_ghost[1] / M  # n1
    P_ghost[2] = U_ghost[2] / M  # n02
    P_ghost[3] = U_ghost[3] / M  # n12

    P_ghost[4] = U_ghost[4] / U_ghost[1]  # U1
    P_ghost[5] = U_ghost[5] / U_ghost[2]  # U02
    P_ghost[6] = U_ghost[6] / U_ghost[3]  # U12
    P_ghost[7] = 2.0 / 3.0 * U_ghost[7] / (phy_const.e * (P_ghost[1]+2*(P_ghost[2]+P_ghost[3])))  # Te
    P_ghost[8] = (P_ghost[1]*P_ghost[4]+2*(P_ghost[2]*P_ghost[5]+P_ghost[3]*P_ghost[6]))/(P_ghost[1]+2*(P_ghost[2]+P_ghost[3])) - J / (A0 * phy_const.e * (P_ghost[1]+2*(P_ghost[2]+P_ghost[3])))  # ve

@njit
def SetOutlet(P_In, U_ghost, P_ghost, J=0.0):

    U_ghost[0] = P_In[0] * M
    U_ghost[1] = P_In[1] * M
    U_ghost[2] = P_In[2] * M
    U_ghost[3] = P_In[3] * M
    U_ghost[4] = P_In[1] * P_In[4] * M
    U_ghost[5] = P_In[2] * P_In[5] * M
    U_ghost[6] = P_In[3] * P_In[6] * M
    U_ghost[7] = 3.0 / 2.0 * (P_In[1] + 2*(P_In[2]+P_In[3])) * phy_const.e * Te_Cath

    P_ghost[0] = U_ghost[0] / M  # ng
    P_ghost[1] = U_ghost[1] / M  # n1
    P_ghost[2] = U_ghost[2] / M  # n02
    P_ghost[3] = U_ghost[3] / M  # n12
    P_ghost[4] = U_ghost[4] / U_ghost[1]  # U1
    P_ghost[5] = U_ghost[5] / U_ghost[2]  # U02
    P_ghost[6] = U_ghost[6] / U_ghost[3]  # U12
    P_ghost[7] = 2.0 / 3.0 * U_ghost[7] / (phy_const.e * (P_ghost[1]+2*(P_ghost[2]+P_ghost[3])))  # Te
    P_ghost[8] = (P_ghost[1]*P_ghost[4]+2*(P_ghost[2]*P_ghost[5]+P_ghost[3]*P_ghost[6]))/(P_ghost[1]+2*(P_ghost[2]+P_ghost[3])) - J / (A0 * phy_const.e * (P_ghost[1]+2*(P_ghost[2]+P_ghost[3])))  # ve


##########################################################
#           Functions defining our numerics              #
##########################################################


# TODO: These are vector. Better allocate them
@njit
def computeMaxEigenVal_e(P):

    U_Bohm = np.sqrt(5 * phy_const.e * P[7, :] / (3 * M))

    return np.maximum(np.abs(U_Bohm - P[8, :]) * 2, np.abs(U_Bohm + P[8, :]) * 2)


@njit
def computeMaxEigenVal_i1(P):

    U_Bohm = np.sqrt(5 * phy_const.e * P[7, :] / (3 * M))

    # return [max(l1, l2) for l1, l2 in zip(abs(U_Bohm - P[2,:]), abs(U_Bohm + P[2,:]))]
    return np.maximum(np.abs(U_Bohm - P[4, :]), np.abs(U_Bohm + P[4, :]))


@njit
def computeMaxEigenVal_i02(P):

    U_Bohm = np.sqrt(5 * phy_const.e * P[7, :] / (3 * M))

    # return [max(l1, l2) for l1, l2 in zip(abs(U_Bohm - P[2,:]), abs(U_Bohm + P[2,:]))]
    return np.maximum(np.abs(U_Bohm - P[5, :]), np.abs(U_Bohm + P[5, :]))


@njit
def computeMaxEigenVal_i12(P):

    U_Bohm = np.sqrt(5 * phy_const.e * P[7, :] / (3 * M))

    # return [max(l1, l2) for l1, l2 in zip(abs(U_Bohm - P[2,:]), abs(U_Bohm + P[2,:]))]
    return np.maximum(np.abs(U_Bohm - P[6, :]), np.abs(U_Bohm + P[6, :]))


@njit
def NumericalFlux(P, U, F_cell, F_interf):

    # Compute the max eigenvalue
    lambda_max_i1_R = computeMaxEigenVal_i1(P[:, 1 : NBPOINTS + 2])
    lambda_max_i1_L = computeMaxEigenVal_i1(P[:, 0 : NBPOINTS + 1])
    lambda_max_i1_star = np.maximum(lambda_max_i1_L, lambda_max_i1_R)

    lambda_max_i02_R = computeMaxEigenVal_i02(P[:, 1 : NBPOINTS + 2])
    lambda_max_i02_L = computeMaxEigenVal_i02(P[:, 0 : NBPOINTS + 1])
    lambda_max_i02_star = np.maximum(lambda_max_i02_L, lambda_max_i02_R)

    lambda_max_i12_R = computeMaxEigenVal_i12(P[:, 1 : NBPOINTS + 2])
    lambda_max_i12_L = computeMaxEigenVal_i12(P[:, 0 : NBPOINTS + 1])
    lambda_max_i12_star = np.maximum(lambda_max_i12_L, lambda_max_i12_R)

    lambda_max_e_R = computeMaxEigenVal_e(P[:, 1 : NBPOINTS + 2])
    lambda_max_e_L = computeMaxEigenVal_e(P[:, 0 : NBPOINTS + 1])
    lambda_max_e_star = np.maximum(lambda_max_e_L, lambda_max_e_R)

    # Compute the flux at the interface

    # Neutrals continuity
    F_interf[0, :] = 0.5 * (
        F_cell[0, 0 : NBPOINTS + 1] + F_cell[0, 1 : NBPOINTS + 2]
    ) - 0.5 * VG * (U[0, 1 : NBPOINTS + 2] - U[0, 0 : NBPOINTS + 1])

    # Ions continuity
    F_interf[1, :] = 0.5 * (
        F_cell[1, 0 : NBPOINTS + 1] + F_cell[1, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i1_star * (U[1, 1 : NBPOINTS + 2] - U[1, 0 : NBPOINTS + 1])
    F_interf[2, :] = 0.5 * (
        F_cell[2, 0 : NBPOINTS + 1] + F_cell[2, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i02_star * (U[2, 1 : NBPOINTS + 2] - U[2, 0 : NBPOINTS + 1])
    F_interf[3, :] = 0.5 * (
        F_cell[3, 0 : NBPOINTS + 1] + F_cell[3, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i12_star * (U[3, 1 : NBPOINTS + 2] - U[3, 0 : NBPOINTS + 1])

    # Ions momentum
    F_interf[4, :] = 0.5 * (
        F_cell[4, 0 : NBPOINTS + 1] + F_cell[4, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i1_star * (U[4, 1 : NBPOINTS + 2] - U[4, 0 : NBPOINTS + 1])
    F_interf[5, :] = 0.5 * (
        F_cell[5, 0 : NBPOINTS + 1] + F_cell[5, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i02_star * (U[5, 1 : NBPOINTS + 2] - U[5, 0 : NBPOINTS + 1])
    F_interf[6, :] = 0.5 * (
        F_cell[6, 0 : NBPOINTS + 1] + F_cell[6, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i12_star * (U[6, 1 : NBPOINTS + 2] - U[6, 0 : NBPOINTS + 1])

    # Electrons
    F_interf[7, :] = 0.5 * (
        F_cell[7, 0 : NBPOINTS + 1] + F_cell[7, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_e_star * (U[7, 1 : NBPOINTS + 2] - U[7, 0 : NBPOINTS + 1])


@njit
def ComputeDelta_t(P):
    # Compute the max eigenvalue
    lambda_max_i1_R = computeMaxEigenVal_i1(P[:, 1 : NBPOINTS + 2])
    lambda_max_i1_L = computeMaxEigenVal_i1(P[:, 0 : NBPOINTS + 1])
    lambda_max_i1_star = np.maximum(lambda_max_i1_L, lambda_max_i1_R)

    lambda_max_i02_R = computeMaxEigenVal_i02(P[:, 1 : NBPOINTS + 2])
    lambda_max_i02_L = computeMaxEigenVal_i02(P[:, 0 : NBPOINTS + 1])
    lambda_max_i02_star = np.maximum(lambda_max_i02_L, lambda_max_i02_R)

    lambda_max_i12_R = computeMaxEigenVal_i12(P[:, 1 : NBPOINTS + 2])
    lambda_max_i12_L = computeMaxEigenVal_i12(P[:, 0 : NBPOINTS + 1])
    lambda_max_i12_star = np.maximum(lambda_max_i12_L, lambda_max_i12_R)

    lambda_max_e_R = computeMaxEigenVal_e(P[:, 1 : NBPOINTS + 2])
    lambda_max_e_L = computeMaxEigenVal_e(P[:, 0 : NBPOINTS + 1])
    lambda_max_e_star = np.maximum(lambda_max_e_L, lambda_max_e_R)

    Delta_t = CFL * Delta_x / (max(max(lambda_max_e_star), max(lambda_max_i1_star), max(lambda_max_i02_star), max(lambda_max_i12_star)))
    return Delta_t


##########################################################################################
#                                                                                        #
#                               SAVE RESULTS                                             #
#                                                                                        #
##########################################################################################

i_save = 0


def SaveResults(P, U, P_Inlet, P_Outlet, J, V, x_center, time, i_save):
    if not os.path.exists(Results):
        os.makedirs(Results)
    ResultsFigs = Results + "/Figs"
    if not os.path.exists(ResultsFigs):
        os.makedirs(ResultsFigs)
    ResultsData = Results + "/Data"
    if not os.path.exists(ResultsData):
        os.makedirs(ResultsData)

    # Save the data
    filenameTemp = ResultsData + "/MacroscopicVars_" + f"{i_save:08d}" + ".pkl"
    pickle.dump(
        [time, P, U, P_Inlet, P_Outlet, J, V, B0, x_center], open(filenameTemp, "wb")
    )  # TODO: Save the current and the electric field


##########################################################################################################
#           Initial field                                                                                #
#           P := Primitive vars [0: ng, 1: ni, 2: ui, 3: Te, 4: ve]                                      #
#           U := Conservative vars [0: rhog, 1: rhoi, 2: rhoiui, 3: 3./2.ni*e*Te]                        #
#                                                                                                        #
##########################################################################################################


time = 0.0
iter = 0
J = 0.0  # Initial Current

# We initialize the primitive variables
P[0, :] *= mdot / (M * A0 * VG)  # Initial propellant density ng
P[1, :] *= NI0*XI1  # Initial n1
P[2, :] *= NI0*XI2_02  # Initial n02
P[3, :] *= NI0*XI2_12  # Initial n12
P[4, :] *= 0.0  # Initial v1
P[5, :] *= 0.0  # Initial v02
P[6, :] *= 0.0  # Initial v12
P[7, :] *= TE0  # Initial Te
P[8, :] *= (P[1, :]*P[4, :] + 2*(P[2, :]*P[5, :]) + 2*(P[3, :]*P[6, :])) - J / (A0 * phy_const.e * (P[1, :]+2*(P[2, :]+P[3, :])))  # Initial Ve

# We initialize the conservative variables
PrimToCons(P, U)


##########################################################################################
#           Loop with Forward Euler                                                      #
#           U^{n+1}_j = U^{n}_j - Dt/Dx(F^n_{j+1/2} - F^n_{j-1/2}) + Dt S^n_j            #
#                                                                                        #
##########################################################################################
print("Starting the simulation with time scheme: ", TIMESCHEME)
temp_count = 0
if TIMESCHEME == "Forward Euler":
    J = compute_I(P, V)
    while time < TIMEFINAL:
        # Save results
        if (iter % SAVERATE) == 0:
            SaveResults(P, U, P_Inlet, P_Outlet, J, V, x_center, time, i_save)
            i_save += 1
            print(
                "Iter = ",
                iter,
                "\tTime = {:.2f}~µs".format(time / 1e-6),
                "\tJ = {:.4f}~A".format(J),
            )

        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet, J, 1)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)

        # Compute the convective Delta t
        Delta_t = ComputeDelta_t(np.concatenate([P_Inlet, P, P_Outlet], axis=1))

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_Inlet, P, P_Outlet], axis=1),
            np.concatenate([U_Inlet, U, U_Outlet], axis=1),
            F_cell,
            F_interf,
        )

        # Compute the source in the center of the cell
        Source(P, S)

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

        time += Delta_t
        iter += 1

if TIMESCHEME == "TVDRK3":

    while time < TIMEFINAL:
        # Save results
        if (iter % SAVERATE) == 0:
            SaveResults(P, U, P_Inlet, P_Outlet, J, V, x_center, time, i_save)
            i_save += 1
            print(
                "Iter = {}".format(iter),
                "\t Time = {:.4f} µs".format(time * 1e6),
                "\t J = {:.4f} A".format(J),
                "\t V = {:.4f} V".format(V),
            )
            if iter == 5:
                sys.exit(1)
        #################################################
        #           FIRST STEP RK3
        #################################################

        # Copy the solution to store it
        U_1[:, :] = U[:, :]
        ConsToPrim(U_1, P_1, J)
        J_1 = compute_I(P, V)
        ConsToPrim(U_1, P_1, J_1)

        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet, J)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)
        # Compute the convective Delta t (Only in the first step)
        Delta_t = ComputeDelta_t(np.concatenate([P_Inlet, P, P_Outlet], axis=1))

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_Inlet, P, P_Outlet], axis=1),
            np.concatenate([U_Inlet, U, U_Outlet], axis=1),
            F_cell,
            F_interf,
        )

        # Compute the source in the center of the cell
        Source(P, S)

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

        # Compute RLC Circuit
        if Circuit:
            dJdt = (J - J0) / Delta_t

            RHS_Volt0[0] = X_Volt0[1]
            RHS_Volt0[1] = (
                -1 / (R * C) * X_Volt0[1] - 1.0 / (L * C) * X_Volt0[0] + 1 / C * dJdt
            )
            X_Volt1 = X_Volt0 + Delta_t * RHS_Volt0

        #################################################
        #           SECOND STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet, J, 2)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_Inlet, P, P_Outlet], axis=1),
            np.concatenate([U_Inlet, U, U_Outlet], axis=1),
            F_cell,
            F_interf,
        )

        # Compute the source in the center of the cell
        Source(P, S)

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
        if Circuit:
            dJdt = (J - J0) / Delta_t
            RHS_Volt1[0] = X_Volt1[1]
            RHS_Volt1[1] = (
                -1 / (R * C) * X_Volt1[1] - 1.0 / (L * C) * X_Volt1[0] + 1 / C * dJdt
            )
            X_Volt2 = 0.75 * X_Volt0 + 0.25 * X_Volt1 + 0.25 * Delta_t * RHS_Volt1

        #################################################
        #           THIRD STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet, J, 3)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_Inlet, P, P_Outlet], axis=1),
            np.concatenate([U_Inlet, U, U_Outlet], axis=1),
            F_cell,
            F_interf,
        )
        # Compute the source in the center of the cell
        Source(P, S)

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
        # Compute the current
        J = compute_I(P, V)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, J)

        # Compute RLC Circuit
        if Circuit:
            dJdt = (J - J0) / Delta_t
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
            J0 = J
            X_Volt0[:] = X_Volt3[:]

            # Change the Voltage
            V = V0 - X_Volt0[0]
        time += Delta_t
        temp_count += 1
        if (iter %SAVERATE) ==0:
            filename = Results + "time_vec_njit.dat"
            ttime_intermediate = ttime.time()
            a_str = " ".join(map(str, [iter, ttime_intermediate - tttime_start]))
            if iter == 0 and os.path.exists(filename):
                os.remove(filename)
                print("File removed:" + filename)
            if os.path.exists(filename):
                with open(filename, 'a') as file:
                    file.write(a_str)
                    file.write("\n")  # Add a newline at the end (optional)
            else:
                with open(filename, 'w') as file:
                    file.write(a_str)
                    file.write("\n")  # Add a newline at the end (optional)
        iter += 1

ttime_end = ttime.time()
print("Exec time = {:.2f} s".format(ttime_end - tttime_start))
