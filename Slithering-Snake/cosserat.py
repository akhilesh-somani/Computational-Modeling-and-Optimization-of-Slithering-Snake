import numpy as np
import numpy.linalg as la

class Cosserat:
    def __init__(self, r_o, Q_o, v_o, omega_o, dt, E, G, A, L, rho, I, dJ, F, Cl, dm, n, boundary, lock_e=True, **kwargs): # 16 inputs
        # Hyperparameters
        self.dt = dt  # time step
        self.lock_e = lock_e # Always set e=1 if set to True
        self.step_idx = 0

        # State Vectors
        self.n = n             # number of elements. n_node = n_elem+1
        self.r_o = r_o         # position vector [n+1,3]
        self.Q_o = Q_o         # frame vector [n,3,3]
        self.v_o = v_o         # velocity vector [n,3]
        self.omega_o = omega_o # ang. velocity vector [n,3]
        # original length and Voronoi domain        
        self.l_hat = self.calc_li(r_o)                            # [n,3]
        self.l_hat_norm = la.norm(self.l_hat, axis=1, keepdims=1) # [n,1]
        self.D_hat = self.calc_Di(self.l_hat_norm)                # [n-1,1]

        # Material Properties
        self.dm = dm   # take the mass calc from validate files [n+1,1]
        # Stiffness Matrix (Assume homogeneous material, Read-Only)
        self.dJ_hat = np.broadcast_to(dJ, (n,3))
        self.S_hat = np.broadcast_to(np.diag([4./3.* G * A, 4./3.* G * A, E * A]), (n,3,3))
        self.B_hat = np.broadcast_to(np.diag([E*I[0], E*I[1], G*I[2]]), (n,3,3))
        # Bending stiffness (Voronoi)
        self.Bi_hat = self.calc_Bi(self.B_hat, self.l_hat_norm, self.D_hat) # [n-1,3,3]

        # Input
        self.F = F               # Force [n+1,3]
        self.Cl = Cl             # Couple [n,3]
        self.boundary = boundary # Only Dirichlet

    def calc_Di(self, l):
        # Calculate Voronoi domain from length
        return (l[1:]+l[:-1])/2

    def calc_li(self, r):
        # Calculate length from position
        return r[1:]-r[:-1]

    def calc_Bi(self, B, l, D):
        # Calculate bending stiffness given Bi, li, Di
        Bdiag = np.diagonal(B, axis1=1, axis2=2)
        B = (Bdiag[1:]*l[1:] + Bdiag[:-1]*l[:-1]) / (2*D)
        return np.array([np.diag(v) for v in B])

    def difference_op(self, x):
        # Calculate differencing operator del
        # First-order accuracy 
        shape = list(x.shape)
        shape[0] += 1
        y = np.zeros(shape)
        y[0] = x[0]
        y[-1] = -x[-1]
        y[1:-1] = x[1:]-x[:-1]
        return y

    def averaging_op(self, x):
        # Calculate averaging operator A
        # Midpoint rule
        shape = list(x.shape)
        shape[0] += 1
        y = np.zeros(shape)
        y[0] = x[0]/2
        y[-1] = x[-1]/2
        y[1:-1] = (x[1:]+x[:-1])/2
        return y

    def rotate_rodrigues(self, omega, theta):
        # Rodrigues rotation to calculate R=exp(theta * e)=exp(-dt/2 omega)
        # theta: [1], omega: [n,3]
        # Calculation is vectorized to account vector R
        '''Normalize'''
        w_norm = la.norm(omega, axis=1)       # Omega norm [n]
        mask = np.isclose(w_norm, 0.0)        # Zero rotation mask [n]
        w_norm[mask] = 0
        theta = (theta * w_norm)[:,None,None] # Angle [n,1,1]
        # Normalize axis of rotation
        u = np.zeros_like(omega)
        u[mask] = omega[mask]
        u[~mask] = omega[~mask] / w_norm[~mask,None]

        '''Skew Symmetrize K: u->K (R^3 -> R^[3x3])'''
        zn = np.zeros(self.n)                                    # Zero vector
        K = np.array([[zn, -u[:,2], u[:,1]],
                      [u[:,2], zn, -u[:,0]],
                      [-u[:,1], u[:,0], zn]]).transpose([2,0,1]) # [n,3,3]

        '''Rotation Matrix'''
        Ksqr = K@K                                       # K^2 [n,3,3]
        I = np.identity(3)[None,:]                       # [1,3,3]
        R = I + np.sin(theta)*K + (1-np.cos(theta))*Ksqr # [n,3,3]
        return R 

    def inverse_rotate(self, t_frameone, t_frametwo):
        # Compute inverse rotation: given two frame, find axis of rotation
        # V = log(t_frametwo, t_frameone)
        assert np.all(t_frameone.shape == t_frametwo.shape)
        n = t_frameone.shape[0] # specify number of batch

        # Find the rotation matrix
        rot_mat = t_frametwo @ t_frameone.transpose([0,2,1]) # [n,3,3]

        # Find the angle by which you rotate using the trace of the matrix
        x = (np.trace(rot_mat, axis1=1, axis2=2) - 1) / 2.0
        x[np.logical_and(np.isclose(x,1.0), x > 1.0)] = 1
        x[np.logical_and(np.isclose(x,-1.0), x > -1.0)] = -1
        if np.any(np.abs(x) > 1.0):
            mask = np.abs(x)>1.0
            raise ValueError("Impossible frame rotation. Double check Q if they are normalized.")
        angle = np.arccos(x) # [n]

        # Find the axis using the logarithm function
        mask = np.isclose(angle, 0)                                   # [n]
        axis_mat = (rot_mat - rot_mat.transpose([0,2,1]))              # [n,3,3]
        axis_mat[mask] = 0
        axis_mat[~mask] = axis_mat[~mask] / (2*np.sin(angle[~mask])[:,None,None])
        ax = np.stack([axis_mat[:,2,1],
                       axis_mat[:,0,2],
                       axis_mat[:,1,0]], axis=1)                   # [n,3]
        ax[~mask] = ax[~mask] / la.norm(ax[~mask], axis=1, keepdims=1)

        return angle[:,None]*ax # [n,3]
    
    def kinematics(self, dt=None):
        # Main stepper function
        # Calculates r, v, Q, omega
        if dt is None:
            dt = self.dt
            theta = -self.dt/2 # Angle of rotation
        else:
            theta = -dt/2

        '''Part 1: Half-step position update'''
        r_half = self.r_o + 0.5 * dt * (self.v_o) # [n+1,3]
        Q_half = self.rotate_rodrigues(self.omega_o, theta) @ self.Q_o # [n,3,3]
        #(TODO)if not np.allclose(Q_half.transpose([0,2,1]) @ Q_half, np.identity(3)[None,:,:]):
        #    raise ValueError('Q is not proper transformation matrix')
        
        '''Part 2: Local acceleration''' 
        lin_acc, ang_acc = self.acceleration(r_half, Q_half) # [n+1,3], [n,3]
        self.v_o = self.v_o + dt * lin_acc # [n+1,3]
        self.omega_o = self.omega_o + dt * ang_acc # [n,3]
        self.boundary_Dirichlet()

        '''Part 3: Second half position update'''
        self.r_o = r_half + 0.5 * dt * self.v_o# [n+1,3]
        self.Q_o = self.rotate_rodrigues(self.omega_o, theta) @ Q_half # [n,3]

        # Impose boundary condition
        self.boundary_Dirichlet()

        return self.get_state
    
    def acceleration(self, r, Q):
        '''State Vectors'''
        # Calculating Tangent Vector
        l = self.calc_li(r)                     # Edge [n,3]
        l_norm = la.norm(l, axis=1, keepdims=1) # Edge length [n,1]
        Di = self.calc_Di(l_norm)               # Voronoi domain [n-1,1]
        ti = l / l_norm                         # Tangent vector (normalized) [n,3]
        if self.lock_e:
            ei = np.ones((self.n,1))
            epsi = np.ones((self.n-1,1))
            epsi_cube = np.ones((self.n-1,1))
        else:
            ei = l_norm / self.l_hat_norm # Dilatation factor [n,1]
            epsi = Di / self.D_hat        # Dilatation factor on Voronoi [n-1,1]
            epsi_cube = epsi**3           # (for simple calculation) [n-1,1]
        
        '''Linear Acceleration'''
        # Calc shear vector
        shear = Q @ (ti*ei - Q[:,2])[:,:,None] # [n,3,1]
        # Calc shear/stretch internal force
        shear_stretch_if = Q.transpose([0,2,1]) @ self.S_hat @ shear        # [n+1,3,1]
        shear_stretch_if = self.difference_op(shear_stretch_if[:,:,0] / ei) # [n+1,3]
        # Calc linear acceleration
        linear_acc = (shear_stretch_if + self.F) / self.dm # [n+1,3]
        
        self._ssif = shear_stretch_if
        self._linacc = linear_acc
        
        '''Angular Acceleration'''
        if self.n == 1:
            # No angular acceleration for single element case
            return linear_acc, np.zeros((1,3))
        # Calculate curvature (kappa_L)
        kappa = (-self.inverse_rotate(Q[:-1], Q[1:]) / self.D_hat)[:,:,None] # [n-1,3,1]
        # Calc bend internal couple
        Bkappa = self.Bi_hat @ kappa                            # [n-1,3,1]
        bend_ic = self.difference_op(Bkappa[:,:,0] / epsi_cube) # [n,3]
        # Calc twist internal couple
        twist_ic = np.cross(kappa, Bkappa, axis=1) # [n-1,3,1]
        twist_ic = twist_ic[:,:,0] * (self.D_hat/epsi_cube)     # [n-1,3]
        twist_ic = self.averaging_op(twist_ic)                  # [n,3]
        # Calc shear/stretch internal couple
        shear_stretch_ic = np.cross(Q@ti[:,:,None], self.S_hat@shear, axis=1) # [n,3,1]
        shear_stretch_ic = shear_stretch_ic[:,:,0] * self.l_hat_norm          # [n,3]
        # Calc angular acceleration
        angular_acc = bend_ic + twist_ic + shear_stretch_ic + self.Cl
        angular_acc = angular_acc / (self.dJ_hat / ei) # [n,3]

        self._kappa = kappa
        self._bic, self._tic, self._ssic = bend_ic, twist_ic, shear_stretch_ic
        self._angacc = angular_acc
        
        return linear_acc, angular_acc

    def boundary_Dirichlet(self):
        # Only Dirichlet boundary condition for r,v,q,w
        for key, boundary in self.boundary.items():
            if key == 'r':
                vector = self.r_o
            elif key == 'v':
                vector = self.v_o
            elif key == 'Q':
                vector = self.Q_o
            elif key == 'w':
                vector = self.omega_o
            else:
                raise KeyError('Wrong boudnary key is found. Only r,v,Q,w are allowed')

            for idx, value in boundary.items():
                vector[idx] = value[self.step_idx] if len(value.shape) > 1 else value
        self.step_idx += 1

    def step(self, dt=None, couple=None, force=None, verbose=False):
        # Change external input to the system : force, couple
        if force is not None:
            assert np.all(self.F.shape == force.shape), "Force shape does not match. Force.shape=[n+1,3]"
            self.F = force
        if couple is not None:
            assert np.all(self.Cl.shape == couple.shape), "Couple shape does not match. Couple.shape=[n,3]"
            self.Cl = couple

        # Run the step
        r,v,q,w = self.kinematics(dt)
        
        # Logging
        if verbose:
            print('r= ', r, '\n\nv=', v, '\n\nq=', q, '\n\nw=', w)

        return r,v,q,w

    @property
    def get_state(self):
        return self.r_o, self.v_o, self.Q_o, self.omega_o

    @property
    def get_tangent(self):
        l = self.calc_li(self.r_o) # Edge [n,3]
        return l

    @property
    def get_shear(self):
        # Return last used shear/stretch internal force
        return self._ssif
#
# if __name__=='__main__':
#     import argparse
#     import matplotlib.pyplot as plt
#     from tqdm import tqdm
#
#     parser = argparse.ArgumentParser(description='Validation Setup')
#     parser.add_argument('-v', '--validate', type=int, help='validation test number')
#     args = parser.parse_args()
#
#     if args.validate == 1:
#         from validate_1 import setup
#         solution = Cosserat(**setup)
#         for _ in range(2):
#             solution.step(verbose=True)
#     elif args.validate == 2:
#         from validate_2 import setup
#         solution = Cosserat(**setup)
#         for _ in range(2):
#             solution.step(verbose=True)
#     elif args.validate == 3:
#         from validate_3 import setup, T, s, y_sol
#         solution = Cosserat(**setup)
#         T = 1
#         t = 0
#         bar = tqdm(total=T)
#         while t < T:
#             bar.update(solution.dt)
#             t += solution.dt
#             r,_,_,_ = solution.step()
#         bar.close()
#
#         fig = plt.figure()
#         plt.plot(r[:,2], r[:,0], label='sim')
#         plt.plot(s, y_sol, label='true')
#         plt.legend()
#         plt.show()
#
#     else:
#         raise ValueError('Incorrect validate argument is passed. Check help [-h]')
