import numpy as np

# Snake
rho = 1e3
E = 1e7
G = 2.0*E/3.0
L = 1
r = 0.025
T_m = 1   # muscular activity period
gamma = 5 # Dissipation constant
n = 50-1  # Number of element. (not vertex)
dt = 2.5e-5

# Friction
mu_kf = 1.019368e4  # Forward kinetic friction coefficient
mu_kb = 1.5*mu_kf # Backward kinetic friction coefficient
mu_sf = 2*mu_kf   # Forward static friction coefficient
mu_sb = 1.5*mu_sf # Backward static friction coefficient
v_eps = 1e-8      # Friction threshold velocity
kw = 1            # Ground stiffness
nu_g = 1e-6       # Ground viscous dissipation

# Material Properties
A = np.pi*r**2 # cross-sectional area
M = A*L*rho
dm = M / n
ds = L / n
dmi = np.ones((n+1,1))*dm
dmi[0] /= 2; dmi[-1] /= 2
I = np.array([(1/4)*np.pi*r**4, (1/4)*np.pi*r**4, (1/2)*np.pi*r**4]) # second moment of Area
dJ = I*rho*ds

# Initial State
r_initial = np.zeros((n+1, 3))
Q_initial = np.zeros((n, 3, 3))
v_initial = np.zeros((n+1, 3))
w_initial = np.zeros((n, 3))

r_initial[:,2] = np.linspace(0, L, n+1) # Uniform length elements
Q_initial[:] = np.identity(3) # snake along k-axis

# External Interaction (Initial)
Force = np.zeros((n+1, 3))
Couple = np.zeros((n,3))

# Boundary Condition
boundary = {
    #'r': { 0: np.array([0., 0., 0.]), -1: np.array([0,0,L]) },
    #'Q': { 0: np.identity(3) },
    #'v': { 0: np.array([0., 0., 0.]) },
    #'w': { 0: np.array([0., 0., 0.]) },
}

# Combine
setup = {
    'r_o'      : r_initial,
    'Q_o'      : Q_initial,
    'v_o'      : v_initial,
    'omega_o'  : w_initial,
    'dt'       : dt,
    'E'        : E,
    'G'        : G,
    'A'        : A,
    'L'        : L,
    'rho'      : rho,
    'I'        : I,
    'dJ'       : dJ,
    'F'        : Force,
    'Cl'       : Couple,
    'dm'       : dmi,
    'n'        : n,
    'mu_kf'    : mu_kf,
    'mu_kb'    : mu_kb,
    'mu_sf'    : mu_sf,
    'mu_sb'    : mu_sb,
    'boundary' : boundary,
    'lock_e'   : False,
}
