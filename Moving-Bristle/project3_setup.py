import numpy as np

# Snake
itr = 200000 #needs to be at least greater than iterations in simulation, used to create moving boundary condition
rho = 1100 # value found from bristle manufacturing material
E = 3.5e8 # value found from bristle manufacturing material
G = 2*E/4000.0
L = 3 #use if using vertical starting position
l = 1
r = .1651# value found from bristle manufacturing material
T_m = 1 # muscular activity period
gamma = 5 # Dissipation constant
n = 4-1 # Number of element. (not vertex)
dt = 2.5e-5

# Friction
mu_kf = 1.019368  # Forward kinetic friction coefficient
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
sq2 = np.sqrt(0.5);
x = np.array([0,0,-sq2,-l-sq2])

#coef = np.array([(2/3)*L/(w**2),(5/3)*L/w,L])
#z = coef[0]*x*x + coef[1]*x + coef[2] #parabolic starting position
z = np.array([0,l,l+sq2,l+sq2])

#x = np.zeros(n+1,) #leave for vertical rod, comment out for rod starting at an angle

r_initial = np.vstack([x.T,np.zeros([n+1,]),z.T]).T  


Q_initial = np.zeros((n, 3, 3))
v_initial = np.zeros((n+1, 3))
w_initial = np.zeros((n, 3))


Q_initial[0] = np.identity(3) # snake along k-axis
theta = np.pi/4
Q_initial[1] = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]) # snake along k-axis
theta = np.pi/2
Q_initial[2] = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])  # snake along k-axis


# External Interaction (Initial)
Force = np.zeros((n+1, 3))
Couple = np.zeros((n,3))

# Boundary Condition

ampl = 4
i = np.linspace(0,itr-1,itr)
#v = ampl * np.cos(i*2*np.pi/itr)*2*np.pi
r = 3*np.linspace(0,itr*dt,itr)

v = 3*np.ones([itr,])*1;
 

R = np.zeros([itr,3])
V = np.zeros([itr,3])     
R[:,0] = r+r_initial[0,0] #position in x 
V[:,0] = v #velocity in x
R2 = np.zeros([itr,3])
R2[:,2] = np.ones([itr,])
R2[:,0] = r+r_initial[1,0]

R3 = np.zeros([itr,3])
R3[:,2] = np.ones([itr,])*(l+sq2)
R3[:,0] = r+r_initial[2,0]


boundary = {
    #'r': { 0: R, 1:R2, 2:R3}, #moving boundary condition (seems to work fine and is not causing the craxy behavior)
    #'v': { 0: V, 1:V, 2:V}, #moving boundary condition
    
    'r': { 0: R, 1:R2}, #moving boundary condition (seems to work fine and is not causing the craxy behavior)
    'v': { 0: V, 1:V}, #moving boundary condition
}

# Combine
#print(r_initial)
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
