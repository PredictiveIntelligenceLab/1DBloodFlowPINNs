
### This is an RCR identification example for identifying RCR parameters for 1D blood flow model

### This code is based on MICHAEL OSTHEGE's code found in 
### https://gist.github.com/michaelosthege/a75b565d3f653721fa235a07eb089912
### for sampling Lorenz attractor.

import abc
import numpy
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from plotting import newfig, savefig

import theano
import theano.tensor as tt
import scipy.integrate
import numpy as np

class Integrator(object):
    """Abstract class of an ODE solver to be used with Theano scan."""
    __metaclass__ = abc.ABCMeta

    def step(self, t, dt, y, dydt, theta):
        """Symbolic integration step.

        Args:
            t (TensorVariable): timepoint to be passed to [dydt]
            dt (TensorVariable): stepsize
            y (TensorVariable): vector of current states
            dydt (callable): dydt function of the model like dydt(y, t, theta)
            theta (TensorVariable): system parameters

        Returns:
            TensorVariable: yprime
        """
        raise NotImplementedError()


class RungeKutta(Integrator):
    def step(self, t, dt, y, dydt, theta):
        k1 = dt*dydt(y, t, theta)
        k2 = dt*dydt(y + 0.5*k1, t, theta)
        k3 = dt*dydt(y + 0.5*k2, t, theta)
        k4 = dt*dydt(y + k3, t, theta)
        y_np1 = y + (1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4
        return y_np1


class TheanoIntegrationOps(object):
    """This is not actually a real Op, but it can be used as if.
    It does differentiable solving of a dynamic system using the provided 'step_theano' method.

    When called, it essentially creates all the steps in the computation graph to get from y0/theta
    to Y_hat.
    """
    def __init__(self, dydt_theano, integrator:Integrator):
        """Creates an Op that uses the [integrator] to solve [dydt].

        Args:
            dydt_theano (callable): function that computes the first derivative of the system
            integrator (Integrator): integrator to use for solving
        """
        self.dydt_theano = dydt_theano
        self.integrator = integrator
        return super().__init__()

    def __step_theano(self, t, y_t, dt_t, t_theta):
        """Step method that will be used in tt.scan.

        Uses the integrator to give a better approximation than dydt alone.

        Args:
            t (TensorVariable): time since intial state
            y_t (TensorVariable): current state of the system
            dt_t (TensorVariable): stepsize
            t_theta (TensorVariable): system parameters

        Returns:
            TensorVariable: change in y at time t
        """
        return self.integrator.step(t, dt_t, y_t, self.dydt_theano, t_theta)

    def __call__(self, y0, theta, dt, n):
        """Creates the computation graph for solving the ODE system.

        Args:
            y0 (TensorVariable or array): initial system state
            theta (TensorVariable or array): system parameters
            dt (float): fixed stepsize for solving
            n (int): number of solving iterations

        Returns:
            TensorVariable: system state y for all t in numpy.arange(0, dt*n) with shape (len(y0),n)
        """
        # TODO: check dtypes, stack and raise warnings
        t_y0 = tt.as_tensor_variable(y0)
        t_theta = tt.as_tensor_variable(theta)

        Y_hat, updates = theano.scan(fn=self.__step_theano,
                                    outputs_info =[{'initial':t_y0}],
                                    sequences=[theano.tensor.arange(dt, dt*n, dt)],
                                    non_sequences=[dt, t_theta],
                                    n_steps=n-1)

        # scan does not return y0, so it must be concatenated
        Y_hat = tt.concatenate((t_y0[None,:], Y_hat))
        # return as (len(y0),n)
        Y_hat = tt.transpose(Y_hat)
        return Y_hat


class InterpolationOps(object):
    """Linearly interpolates the entries in a tensor according to vectors of
    predicted and desired coordinates.
    """
    def __init__(self, x_predicted, x_interpolated):
        """Prepare an interpolation subgraph.

        Args:
            x_predicted (ndarray): x-coordinates for which Y will be predicted (T_pred,)
            x_interpolated (ndarray): x-coordinates for which Y is desired (T_data,)
        """
        assert x_interpolated[-1] <= x_predicted[-1], "x_predicted[-1]={} but " \
            "x_interpolated[-1]={}".format(x_predicted[-1], x_interpolated[-1])
        self.x_predicted = x_predicted
        self.x_interpolated = x_interpolated
        self.weights = tt.as_tensor_variable(
            interpolation_weights(x_predicted, x_interpolated))
        return super().__init__()

    def __call__(self, Y_predicted):
        """Symbolically apply interpolation.

        Args:
            Y_predicted (ndarray or TensorVariable): predictions at x_pred with shape (N_Y,T_pred)

        Returns:
            Y_interpolated (TensorVariable): interpolated predictions at x_data with shape (N_Y,T_data)
        """
        Y_predicted = tt.as_tensor_variable(Y_predicted)
        Y_interpolated = tt.dot(self.weights, tt.transpose(Y_predicted))
        return tt.transpose(Y_interpolated)


if __name__ == '__main__':
    vessel_1 = np.load("results_real.npy").item()
    
    t_np = vessel_1["Time"]
    p_ref = vessel_1["Pressure_2"][:,-1]
    y = vessel_1["Flow_2"][:,-1]
    
    t_np = t_np - t_np.min()
    T = t_np.max() 
    p_exact = p_ref
    p = p_exact[:,None]
    
    t = t_np

    NN = y.shape[0]
    
    def get_fft(y, T):
        n = y.shape[0] - 1
        yy = y[0:n]
        N = n
        mN = N//2
        c = np.fft.fft(yy, N)
        aa = 2*np.real(c[0:mN+1])/N
        bb = -2*np.imag(c[0:mN+1])/N
        return aa, bb

    a_global, b_global = get_fft(y, T)

    def compute_y(t, a, b, nmodes, T):
        output = 0.5*a[0]
        diff = 0
        for i in range(1, nmodes):
            output = output + a[i]*np.cos(2*np.pi*i*t/T) + b[i]*np.sin(2*np.pi*i*t/T)
            diff = diff - 2*np.pi*i/T*a[i]*np.sin(2*np.pi*i*t/T) + 2*np.pi*i/T*b[i]*np.cos(2*np.pi*i*t/T)
        return output, diff


    def interpolation_weights(x_predicted, x_interpolated):
        """Computes weights for use in left-handed dot product with y_fix.

        Args:
            x_predicted (numpy.ndarray): x-values at which Y is predicted
            x_interpolated (numpy.ndarray): x-values for which Y is desired

        Returns:
            weights (numpy.ndarray): weights for Y_desired = dot(weights, Y_predicted)
        """
        x_repeat = numpy.tile(x_interpolated[:,None], (len(x_predicted),))
        distances = numpy.abs(x_repeat - x_predicted)

        x_indices = numpy.searchsorted(x_predicted, x_interpolated)

        weights = numpy.zeros_like(distances)
        idx = numpy.arange(len(x_indices))
        weights[idx,x_indices] = distances[idx,x_indices-1]
        weights[idx,x_indices-1] = distances[idx,x_indices]
        weights /= numpy.sum(weights, axis=1)[:,None]
        return weights



    def dydt(y, t, theta):

        q, q_diff = compute_y(t, a_global, b_global, 50, T)

        R1, R2, C, Pinf = theta
        yprime = -y/R2/C + (R1 + R2)/R2/C * q + Pinf/R2/C + R1 * q_diff

        return yprime

    def dydt_theano(y, t, theta):
        q, q_diff = compute_y(t, a_global, b_global, 50, T)
        # get parameters
        R1 = theta[0]
        R2 = theta[1]
        C = theta[2]
        Pinf = theta[3]
        # set up differential equations
        yprime = tt.zeros_like(y)
        yprime = tt.set_subtensor(yprime[0], -y/R2/C + (R1 + R2)/R2/C * q + Pinf/R2/C + R1 * q_diff)
        return yprime


    x = t.flatten()

    def compute_norm(x, x_pred):
        return np.sqrt(np.sum((x-x_pred)**2))/np.sqrt(np.sum(x**2))
    
    def search_params(R, C):
        counter = 0
        error = np.zeros((R.shape[0], C.shape[0]))
        for i in range(R.shape[0]):
            for j in range(C.shape[0]):
                truth = numpy.array([1.931814e+08 , R[i,j], C[i,j], 666.5])
                p_pred = scipy.integrate.odeint(dydt, p_exact[0], x, (truth,))
                error[i,j] = compute_norm(p, p_pred)
                counter = counter + 1
                print('Sample %d/%d' % (counter, nn**2))
        idx_row, idx_col = np.where(error == np.amin(error[np.nonzero(error)]))
        best_R, best_C = R[idx_row,idx_col], C[idx_row,idx_col]
        return best_R, best_C, error
        
    print(x.shape)
    
    # Initial coarse search
    nn = 10
    lb = np.array([19.0, -19.0])
    ub = np.array([29.0, -29.0])
    R_prop = np.exp(np.linspace(lb[0], ub[0], nn))
    C_prop = np.exp(np.linspace(lb[1], ub[1], nn))
    R, C = np.meshgrid(R_prop, C_prop)
    best_R, best_C, error = search_params(R, C)
    print('Initial search: R: %e, C: %e' % (best_R, best_C))
    
    # Adaptively refined search
    num_iters = 5
    for i in range(0, num_iters):
        lb = np.array([best_R - 0.5*best_R, best_C - 0.5*best_C])
        ub = np.array([best_R + 0.5*best_R, best_C + 0.5*best_C])
        R_prop = np.linspace(lb[0], ub[0], nn)
        C_prop = np.linspace(lb[1], ub[1], nn)
        R, C = np.meshgrid(R_prop, C_prop)
        best_R, best_C, error = search_params(R, C)
        print('Iteration %d/%d: R: %e, C: %e' % (i+1, num_iters, best_R, best_C))
    
    idx_row, idx_col = np.where(error == np.amin(error[np.nonzero(error)]))
    prediction = numpy.array([1.931814e+08 , best_R, best_C, 666.5])
    p_pred = scipy.integrate.odeint(dydt, p_exact[0], x, (prediction,))
    print(1.931814e+08 , best_R, best_C)
    
    plt.plot(x, p_pred/133., 'r-')
    plt.plot(x, p_exact/133., 'b--')
    
    fig = plt.figure(figsize=(8, 6), dpi=500, facecolor='w', frameon = False)
    fig.clf()
    ax = fig.gca(projection='3d')


    surf = ax.plot_surface(R, C, error, cmap=cm.coolwarm,
                       linewidth=1, alpha = 0.8)
    
    fig.colorbar(surf, shrink=0.8, aspect=10).ax.tick_params(labelsize=11)

    plt.plot(best_R, best_C, error[idx_row,idx_col], color='r', alpha=1, marker='*',markersize=8)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.view_init(50, 60)
    
    fig.savefig("error_surface.png")
    
    dictionary = {}
    dictionary = {"R": R,
                  "C": C,
                  "Error": error
                  }
    
    np.save("RCR_real_2",dictionary)
    