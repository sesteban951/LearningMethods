
# standard includes
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

##################################################################################
# ODE SOLVER
##################################################################################

class ODESolver:
    """
        ODE Solver Class
    """
    # initialization
    def __init__(self, dynamics, method):
        
        # dynamics to propagate 
        self.dynamics = dynamics

        # method to use for integration
        self.method = method

        # print message
        print(f"ODESolver initialized with dynamics: {self.dynamics.__class__.__name__} and method: {self.method}")

    # local dynamics function 
    def ode_func(self, t, x_flat):

        # vecotrize the state
        x = x_flat.reshape((self.dynamics.nx, 1))

        # compute control action
        u = self.dynamics.k(t, x)

        # compute the dynamics
        x_dot = self.dynamics.f(x, u)

        return x_dot.flatten()

    # function to propagate the dynamics for a single initial condition
    def fwd_propagate(self, x0, dt, N):
        
        # make sure the shape of x0 is correct
        assert x0.shape == (self.dynamics.nx, 1), "Initial state x0 must be of shape (nx, 1)"

        # initialize the state solution
        x_traj = np.zeros((self.dynamics.nx, N + 1)) # (nx, N + 1)
        t_traj = np.arange(N + 1) * dt               # (N+1, )

        # populate the initial state
        x_traj[:, 0] = x0.flatten()

        # loop over each step and integrate
        for k in range(N):

            # initial and final times
            t0 = t_traj[k]
            tf = t_traj[k + 1]

            # integrate one step 
            sol = sp.integrate.solve_ivp(self.ode_func,
                                         (t0, tf),
                                         x_traj[:, k],
                                         method=self.method,
                                         t_eval=[tf])
            
            # store the state
            x_traj[:, k + 1] = sol.y[:,-1]

        return t_traj, x_traj
    
    # forward propogation with multiple initial conditions
    def fwd_propagate_multiple(self, X0, dt, N):

        # make sure the X0 array is the right size
        X0_rows = X0.shape[0]
        X0_cols = X0.shape[1]
        assert X0_rows == self.dynamics.nx, "X0 must have shape (nx, num_conditions)"

        # initialize the state solution list
        X_traj = np.zeros((self.dynamics.nx, N + 1, X0_cols))   # (nx, N + 1, num_conditions)
        t_traj = np.arange(N + 1) * dt                          # (N+1, )

        # loop over each initial condition
        for i in range(X0_cols):
            
            # propagate the dynamics for each initial condition
            _, x_traj = self.fwd_propagate(X0[:, i].reshape((self.dynamics.nx, 1)), dt, N)

            # store the results
            X_traj[:, :, i] = x_traj

        return t_traj, X_traj
        

##################################################################################
# DYNAMICS
##################################################################################

class DoubleIntegrator:
    """
    Double Integrator Dynamics Class
    """
    # initialization
    def __init__(self):

        # state and input dimensions
        self.nx = 2
        self.nu = 1

        # system parameters
        self.A = np.array([[0, 1], 
                           [0, 0]])
        self.B = np.array([[0], 
                           [1]])
        
        # gains
        self.kp = 10.0
        self.kd = 1.0

    # dynamics function
    def f(self, x, u):
        """
            ẋ = Ax + Bu
            where A = [[0, 1],   B = [[0],
                       [0, 0]]        [1]]
        """
        # make sure the shape of x and u is correct
        assert x.shape == (self.nx, 1), "State x must be of shape (nx, 1)"
        assert u.shape == (self.nu, 1), "Input u must be of shape (nu, 1)"

        # compute the dynamics
        x_dot = self.A @ x + self.B @ u

        return x_dot
        
    # feedback controller
    def k(self, t, x):
        """
            Simple proportional controller
            u = -Kx
            where K = [[1, 0]]
        """
        # make sure the shape of x is correct
        assert x.shape == (self.nx, 1), "State x must be of shape (nx, 1)"

        # controller gain
        K = np.array([[self.kp, self.kd]])

        # compute the control input
        u = -K @ x

        return u.reshape((self.nu, 1))


##################################################################################
# TESTING
##################################################################################

if __name__ == "__main__":
    
    # Example usage
    dynamics = DoubleIntegrator()

    # desired integration 
    method = 'RK45'  # or 'RK23', 'DOP853', etc.
    ode_solver = ODESolver(dynamics, method)

    # solve a single initial condition
    x0 = np.array([[1], [1]])  # initial state
    dt = 0.01                  # time step
    N = 500                    # number of steps

    # uniformly sample initial conditions
    N_samples = 50
    X0 = np.zeros((dynamics.nx, N_samples))
    for i in range(N_samples):

        # randomly sample initial conditions
        x0 = np.random.uniform(-1, 1, (dynamics.nx, 1))

        # store the initial condition
        X0[:, i] = x0.flatten()

    # plot the results
    # t_traj, x_traj = ode_solver.fwd_propagate(x0, dt, N)
    # plt.plot(x_traj[0, :], x_traj[1, :], label='Position')

    # plot 
    t_traj, X_traj = ode_solver.fwd_propagate_multiple(X0, dt, N)
    for i in range(X_traj.shape[2]):
        plt.plot(X_traj[0, :, i], X_traj[1, :, i], label=f'Condition {i+1}')

    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (v)')
    plt.title('Double Integrator Dynamics')
    plt.show()
