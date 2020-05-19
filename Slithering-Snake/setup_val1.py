import numpy as np

# Spatial Discretization
# Divide into n parts

L = 3. # Length along z-axis
n = 1 # Number of elements
rho = 5000
r = 0.25
E = 1e6
dt = 3e-4
G = 1e4
A = np.pi * (r) ** 2
M = A * L * rho
mi = M / n
ds = L / n
I = np.array([(1/4)*np.pi*r**4, (1/4)*np.pi*r**4, (1/2)*np.pi*r**4]) # second moment of Area
#dJ = np.array([mi*(L/n)**2/12, mi*(L/n)**2/12, mi*r**2/2])
dJ = I * rho * ds


# Initial State
r_initial = np.zeros((n+1, 3))
Q_initial = np.zeros((n, 3, 3))
v_initial = np.zeros((n+1, 3))
omega_initial = np.zeros((n, 3))
Q_initial[:] = np.identity(3)

r_initial[:,2] = np.linspace(0, L, n+1)

mass = np.ones((n+1,1))*mi
mass[0] /= 2; mass[-1] /= 2

f = 15.0
F = np.zeros((n+1, 3))
F[-1] = np.array([0.,0.,f])
boundary = {
    'r': { 0: np.array([0., 0., 0.]) },
    'v': { 0: np.array([0., 0., 0.]) },
}
setup = {
    'r_o': r_initial,
    'Q_o': Q_initial,
    'v_o': v_initial,
    'omega_o': omega_initial,
    'dt': dt,
    'E' : E,
    'G' : G,
    'A' : A,  # Area
    'rho' : rho,  # density
    'F' : F,  # force on each node (downwards)
    'Cl': np.zeros((n,3)),
    'L' : L,
    'I' : I,
    'dJ': dJ,
    'dm' : mass,
    'n' : n,
    'boundary' : boundary,
    'lock_e' : False
}
L_true = ((f/(A*E)) * L) + L
