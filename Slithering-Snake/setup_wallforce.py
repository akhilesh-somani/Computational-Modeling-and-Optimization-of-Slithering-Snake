import numpy as np

case = 2 # Case numbers [1, 2, 3, 4]

# Spatial Discretization
# Divide into n parts
L = 3. # Length along z-axis
n = 100 # Number of elements
iterations = 3
rho = 5000
r = 0.025
A = np.pi * (r) ** 2
M = A * L * rho
I = np.array([(1/4)*M*r**2, (1/4)*M*r**2, (1/2)*M*r**2])
mi = M / n
dJ = np.array([mi*(L/n)**2/12, mi*(L/n)**2/12, mi*r**2/2])

# Assumption for penetration calc: Wall is at: r (in the negative y - direction)
# So an initial r value of 0 means that the rod is just touching the wall beneath it.
r_initial = np.zeros((n+1, 3))
Q_initial = np.zeros((n, 3, 3))
v_initial = np.zeros((n+1, 3))
omega_initial = np.zeros((n, 3))

Q_initial[:] = np.identity(3)
r_initial[:,2] = np.linspace(0, L, n+1)

mass = np.ones((n+1,1))*mi
mass[0] /= 2; mass[-1] /= 2

mass = np.array(mass)
F = np.zeros((n+1, 3))

# Wall-Modeling Test Cases are performed using n = 1 element initially
if case == 1: # No gravity + Only penetration in the wall
    r_initial[:, 1] = - r # All elements are penetrated by a distance of r

elif case == 2: # Only gravity + no initial penetration for the element
    F[:, 1] = - np.squeeze(mass) * 9.81 # Gravitational force on each node
    r_initial[:, 1] = r # Initialize the rod from some height

elif case == 3:
    v_initial[0] = [2., 0., -9.]
    v_initial[1] = [3., 0., -19.]
    v_initial[n-1] = [-2., 0., 3.]
    v_initial[-1] = [0., 0., -5.]
elif case == 4:
    v_initial[0] = [2., 1., -9.]
    v_initial[1] = [3., 2., -19.]
    v_initial[n - 1] = [-2., -1., 3.]
    v_initial[-1] = [0., -2., -5.]

setup = {
    'r_o': r_initial,
    'Q_o': Q_initial,
    'v_o': v_initial,
    'omega_o': omega_initial,
    'dt': 3e-4,
    'iterations': iterations,
    'E' : 1e6,  # E
    'G' : 1e4,  # G
    'A' : A,  # Area
    'rho' : rho,  # density
    'F' : F,  # force on each node
    'Cl': np.zeros((n,3)),
    'L' : L,
    'I' : I,
    'dm' : mass,
    'n' : n,
    'M' : M,
    'dJ' :dJ,
    'boundary' : {}
}

