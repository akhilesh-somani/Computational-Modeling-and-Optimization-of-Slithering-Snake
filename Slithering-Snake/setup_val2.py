import numpy as np

# Spatial Discretization
# Divide into n parts

L = 3. # Length along z-axis
n = 7 # Number of elements
r_initial = []

rho = 5000
r = 0.25
A = np.pi * (r) ** 2
M = A * L * rho
mi = M / n
ds = L / n
I = np.array([(1/4)*np.pi*r**4, (1/4)*np.pi*r**4, (1/2)*np.pi*r**4]) # second moment of Area
h = L/n
dJ = I*ds*rho
dJ = np.array([mi*(L/n)**2/12, mi*(L/n)**2/12, mi*r**2/2])

mass = np.ones((n+1,1))*mi
mass[0] /= 2; mass[-1] /= 2

Q_initial = np.zeros((n, 3, 3))
v_initial = np.zeros((n+1, 3))
omega_initial = np.zeros((n, 3))
Q_initial[:] = np.identity(3)

# initialize r, v, q, w
for i in range(n + 1):
    r_initial.append([0. ,4*((1/4) - ((1/n)*(i-(n/2)))**2), i * ( L / n)])

r_initial = np.array(r_initial)
F = np.zeros((n+1, 3))
#F[-1] = [0.,1,0]
#F[0] = [0,1,0]
boundary = {
    'r': { 0: np.array([0., 0., 0.]),-1: np.array([0., 0., L]) },
    'v': { 0: np.array([0., 0., 0.]),-1: np.array([0., 0., 0.])  },
}
setup = {
    'r_o': r_initial,
    'Q_o': Q_initial,
    'v_o': v_initial,
    'omega_o': omega_initial,
    'dt': 3e-4,
    'E' : 1e6,  # E
    'G' : 1e4,  # G
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
    'lock_e': False
}
