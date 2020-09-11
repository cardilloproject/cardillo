from cardillo.utility.coo import Coo
from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.solver import Solution

import numpy as np
from scipy.sparse.linalg import spsolve # SuperLU direct solver
from scipy.sparse.linalg import splu # SuperLU direct solver
from scipy.sparse.linalg import LinearOperator, spilu # incomplete LU preconditioner
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, lgmres, minres, qmr, gcrotmk # iterative solvers
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import bmat
from tqdm import tqdm

# raise exceptions for MatrixRankWarning of scipy
from scipy.sparse.linalg.dsolve.linsolve import MatrixRankWarning as MatrixRankWarning
import warnings
warnings.filterwarnings('error', category=MatrixRankWarning)

class Newton():
    r"""Force and displacement controlled Newton--Raphson method. This solver is used to find a static solution for a mechanical system.

    Parameters
    ----------
    model : :class:`Model<cardillo.model>` or :class:`SparseModel<cardillo.model>`
        Model or SparseModel object. Determines representation of matrix objects.
    n_load_steps : int
        number of load steps (ignored if load_steps is not None)
    load_steps : numpy.ndarray
        list of load steps within the interval :math:`[0,1]`. If initialized with *None*, ``load_steps = [1/n_load_steps, 2/n_load_steps,...,1]`` is used.
    tol : float
        error tolerance.
    max_iter : int
        maximum number of iterations.

    Notes
    -----
    Let the following be given functions of the load step parameter (time) :math:`t`, generalized coordinates :math:`\mathbf{q}` and 
    generalized velocity :math:`\mathbf{u}`.
    
    - :math:`\mathbf{f}^\mathrm{int}(t,\mathbf{q},\mathbf{u})` the internal forces of all bodies which constitute the mechanical system 
    - :math:`\mathbf{f}^\mathrm{ext}(t,\mathbf{q},\mathbf{u})` the forces describing all force laws acting on the system
    - :math:`\mathbf{g}(t,\mathbf{q})` the gap function describing the bilateral constraints
    - :math:`\mathbf{W}^\textrm{T}=\frac{\partial\mathbf{g}}{\partial\mathbf{q}}` the generalized force directions of the constraints
    
    For every load step :math:`0\leq t_1\leq \dots \leq t_i \leq \dots \leq 1` the nonlinear equation 

    .. math::

        \begin{equation}
        \mathbf{R}(\mathbf{q}_i,\mathbf{\lambda}_i) =
        \begin{pmatrix}
        \mathbf{f}^\mathrm{int}(0,\mathbf{q}_i,0) + \mathbf{f}^\mathrm{ext}(t_i,\mathbf{q}_i,0) + \mathbf{W}(t_i,\mathbf{q}_i)\mathbf{\lambda}_i\\
        \mathbf{g}(t_i,\mathbf{q}_i)
        \end{pmatrix} = 0
        \end{equation}
    
    is solved for the generalized positions :math:`\mathbf{q}_i` and the constraint force vector :math:`\mathbf{\lambda}_i.` 
    The solution is found using the Newton--Raphson method which is defined as follows:

    The method starts with an initial guess :math:`(\mathbf{q}^0_i,\mathbf{\lambda}^0_i)` which is the initial values provided by the model 
    if :math:`i = 1` (first load step) and is the solution of the previous load step otherwise. The update formula
    
    .. math::
        \begin{equation}
        \begin{pmatrix}
        \mathbf{q}^{k+1}_i\\
        \mathbf{\lambda}^{k+1}_i
        \end{pmatrix} = 
        \begin{pmatrix}
        \mathbf{q}^{k}_i\\
        \mathbf{\lambda}^{k}_i
        \end{pmatrix}
        -
        \left.\left(
        \frac{\partial \mathbf{R}}{\partial \mathbf{q}_i}\quad  \frac{\partial \mathbf{R}}{\partial \mathbf{\lambda}_i} 
        \right)\right|^{-1}_{(\mathbf{q}^k_i,\mathbf{\lambda}^k_i)}
        \begin{pmatrix}
        \mathbf{q}^{k}_i\\
        \mathbf{\lambda}^{k}_i
        \end{pmatrix}
        \end{equation}

    is applied until :math:`||\mathbf{R}(\mathbf{q}^k_i,\mathbf{\lambda}^k_i)||_\infty < \mathrm{tol}`, where :math:`||\,.\,||_\infty` is the maximum norm, or the maximum number of iterations is reached (:math:`k=\mathrm{maxIter}`).

    Attention
    ---------

    External forces and gap functions are incremented in each load step if they are depending on a parameter `t`. So force controlled Netwon is obtained by constructing a time constant gap function. On the other hand a displacement controlled Newton method is obtained by passing a constant external forces and time dependent gap functions.
    
    """
        
    def _residual(self, t, x):
        nq = self.nq
        q = x[:nq]
        la = x[nq:]

        R = np.zeros(self.nx)
        self.W_g = self.model.W_g(t, q)
        R[:nq] = self.model.h(t, q, self.u) + self.W_g @ la
        R[nq:] = self.model.g(t, q)

        return R

    def _jacobian_num(self, t, x, scipy_matrix=csc_matrix):
        return scipy_matrix(Numerical_derivative(self._residual, order=2)._x(t, x))
        
    def _jacobian_an(self, t, x):
        nq = self.nq
        q = x[:nq]
        la = x[nq:]

        K = self.model.h_q(t, q, self.u) + self.model.Wla_g_q(t, q, la)
        g_q = self.model.g_q(t, q)

        return bmat([[K,   self.W_g], \
                     [g_q,     None]], format='csc')
    
    def __init__(self, model, n_load_steps=1, load_steps=None, tol=1e-8, max_iter=50, sparse_solver='scipyLU', iterative_tol=1.0e-10, numerical_jacobian=False, error_function=None):
        self.max_iter = max_iter
        self.tol = tol
        self.model = model

        if load_steps is None:
            self.load_steps = np.linspace(0, 1, n_load_steps)
        else:
            self.load_steps = np.array(load_steps)

        # dimensions
        self.nt = len(self.load_steps)
        self.nq = self.model.nq
        self.nla = self.model.nla_g
        self.nx = self.nq + self.nla
        
        # memory allocation
        self.x = np.zeros((self.nt, self.nx))

        # initial conditions
        self.x[0] = np.concatenate((self.model.q0, self.model.la_g0))
        self.u = np.zeros(self.nq) # zeros as system is static

        # chose sparse solver
        sparse_direct_solvers = {'superLU': lambda A, b: splu(A).solve(b),
                                 'scipyLU': lambda A, b: spsolve(A, b),
                                } 
        sparse_iterative_solvers = {'cgs': lambda A, b, M:       cgs(A, b, M=M, tol=iterative_tol),
                                    'gmres': lambda A, b, M:     gmres(A, b, M=M, tol=iterative_tol),
                                    'lgmres': lambda A, b, M:    lgmres(A, b, M=M, tol=iterative_tol),
                                    'gcrotmk': lambda A, b, M:   gcrotmk(A, b, M=M, tol=iterative_tol),
                                    'bicgstab': lambda A, b, M:  bicgstab(A, b, M=M, tol=iterative_tol),
                                   }
        if sparse_solver in sparse_direct_solvers:
            self.direct_solver = True
            self.sparse_solver = sparse_direct_solvers[sparse_solver]
        elif sparse_solver in sparse_iterative_solvers:
            self.direct_solver = False
            self.sparse_solver = sparse_iterative_solvers[sparse_solver]
        else:
            raise ValueError(f'sparse solver: {sparse_solver} is not supported')

        if numerical_jacobian:
            self._jacobian = self._jacobian_num
        else:
            self._jacobian = self._jacobian_an

        if error_function is None:
            self.error_function = lambda R: np.absolute(R).max()
        else:
            self.error_function = error_function
        
    def solve(self):
        # compute numbe rof digits for status update
        len_t = len(str(self.nt))
        len_maxIter = len(str(self.max_iter))

        pbar = tqdm(range(0, self.nt), leave=True)
        for i in pbar:
            pbar.update(1)
            
            # compute initial residual
            R = self._residual(self.load_steps[i], self.x[i])
            error = self.error_function(R)
            
            # reset counter and print inital status
            k = 0
            pbar.set_description(f' force iter {i+1:>{len_t}d}/{self.nt}; Newton steps {k:>{len_maxIter}d}/{self.max_iter}; error {error:.4e}/{self.tol:.2e}')
            
            # perform netwon step if necessary
            if error >= self.tol:
                while k <= self.max_iter:
                    # compute jacobian
                    dR = self._jacobian(self.load_steps[i], self.x[i])

                    # solve linear system of equations
                    if self.direct_solver:
                        update = self.sparse_solver(dR, R)
                    else:
                        # preconditioner
                        ilu = spilu(dR)
                        Mx = lambda x: ilu.solve(x)
                        M = LinearOperator((self.nx, self.nx), Mx)

                        # solve linear system
                        update = self.sparse_solver(dR, R, M)[0]

                    # perform update
                    self.x[i] -= update
                                
                    # compute new residual
                    R = self._residual(self.load_steps[i], self.x[i])
                    error = self.error_function(R)

                    # update counter and print status
                    k += 1
                    pbar.set_description(f' force iter {i+1:>{len_t}d}/{self.nt}; Newton steps {k:>{len_maxIter}d}/{self.max_iter}; error {error:.4e}/{self.tol:.2e}')

                    # check convergence
                    if (error < self.tol):
                        break

            if k > self.max_iter:
                # return solution up to this iteration
                pbar.close()
                print(f'Newton-Raphson method not converged, returning solution up to iteration {i+1:>{len_t}d}/{self.nt}')
                return Solution(t=self.load_steps, q=self.x[:i+1, :self.nq], la_g=self.x[:i+1, self.nq:])
                    
            # store solution as new initial guess
            if i < self.nt - 1:
                self.x[i + 1] = self.x[i]
                
        # return solution object
        pbar.close()
        return Solution(t=self.load_steps, q=self.x[:, :self.nq], la_g=self.x[:, self.nq:])
