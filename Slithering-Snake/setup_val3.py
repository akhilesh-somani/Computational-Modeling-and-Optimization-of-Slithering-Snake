import numpy as np

# Timoshenko beam parameters (SI units)
rho = 5e3
E = 1e6
G = 1e4
F = np.array([-15,0,0])
L = 3
r = 0.25
gamma = 0.1
T = 5000
n = 100-1 # Number of element. (not vertex)
dt = 3e-4

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

r_initial[:,2] = np.linspace(0, L, n+1)
Q_initial[:] = np.identity(3)

# External Interaction
Force = np.zeros((n+1, 3))
Force[-1] = F
Couple = np.zeros((n,3))

# Boundary Condition
boundary = {
    'r': { 0: np.array([0., 0., 0.]) },
    'Q': { 0: np.identity(3) },
    'v': { 0: np.array([0., 0., 0.]) },
    'w': { 0: np.array([0., 0., 0.]) },
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
    'boundary' : boundary,
    'lock_e'   : False,
}
s = np.linspace(0,L,n+1)
y_sol = (3*F[0]/(4*A*G))*s - (-F[0]*L/(2*E*I[0]))*(s**2) + (-F[0]/(6*E*I[0]))*(s**3)
