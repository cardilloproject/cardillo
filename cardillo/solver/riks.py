from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.solver import Solution

import numpy as np
from scipy.sparse.linalg import spsolve # SuperLU direct solver
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import bmat
from tqdm import tqdm

# TODO: generalize this for arbitrary arc-length functions
# TODO: try to build arc-length equations that measures the external and constraint forces
# TODO: adapt Crisfield notation? la: load level, q_ef: fixed external loading vector
# TODO. cite original works of Riks and Wempner, see Crisfield1991
# TODO: predictor solution, see Crisfield1991 section 9.4.3
# TODO: automatic increment cutting: Crisfield1991 section 9.5.1
# TODO: read Crisfield1996 section 21 Branch switching and further advanced solution procedures
#       - line-searches with Riks/Wempner linear arc-length method Crisfield1996 section 21.7.1
  
class Riks():
    r"""Linear arc-length solver close to Riks method as dervied in Crisfield1991 section 9.3.2 
    p.273. A variable arc-length is chosen as shown by Crisfield1981 or Crisfield 1983. For the 
    first predictor a tangent predictor is used. For all other predictors a simple secant predictor
    is used. This enables the solver to 'run forward' instead of 'doubling back on its track'.

    References
    ----------
    Crisfield1991: http://freeit.free.fr/Finite%20Element/Crisfield%20M.A.%20Vol.1.%20Non-Linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Essentials%20(Wiley,19.pdf
    Crisfield1997: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    # def __f_Wriggers(self, la_arc, x, ds, psi):
    #     dx = x - self.x_i
    #     dla_arc = la_arc - self.la_arc_i

    #     # Wriggers (Tschöpe)
    #     return np.sqrt(dx @ dx + dla_arc * dla_arc) - ds

    # def __f_Crisfield(self, la_arc, x, ds, psi):
    #     # extract generalized coordinates and lagrange multipliers
    #     nq = self.nq
    #     q = x[:nq]

    #     dx = x - self.x_i
    #     dla_arc = la_arc - self.la_arc_i

    #     # Crisfield original
    #     fe = self.model.forceLaws(1, q, np.zeros_like(q))
    #     c = dx @ dx + dla_arc * dla_arc * psi**2 * (fe @ fe)
    #     return c - ds**2        
        
    # def __df_Wriggers(self, la_arc, x, ds, psi):

    #     dx = x - self.x_i
    #     dla_arc = la_arc - self.la_arc_i

    #     # Wriggers (Tschöpe)
    #     c = np.sqrt(dx @ dx + dla_arc * dla_arc)
    #     df_dx = 1 / c * dx
    #     df_dla = 1 / c * dla_arc
    #     return df_dx, df_dla
        
    # def __df_Crisfield(self, la_arc, x, ds, psi):
    #     # extract generalized coordinates
    #     nq = self.nq
    #     q = x[:nq]

    #     dx = x - self.x_i
    #     dla_arc = la_arc - self.la_arc_i
        
    #     # Crisfield original
    #     fe = self.model.forceLaws(1, q, np.zeros_like(q))
    #     df_dx = 2 * dx
    #     df_dla = 2 * dla_arc * psi**2 * (fe @ fe)
    #     return df_dx, df_dla

    def __f_simple(self, x, ds, psi):
        """TODO: cite equation number in Crisfield1991.
        """
        # limit generalized coordinates only
        nq = self.nq
        dq = x[:nq] - self.x_i[:nq]
        return dq @ dq[:nq] - ds * ds
        
        # # limit generalized coordinates and Lagrange-multipliers
        # nq = self.nq
        # nla = self.nla
        # dqla = x[:nq + nla] - self.x_i[:nq + nla]
        # return dx @ dx - ds * ds
        
    def __df_simple(self, x, ds, psi):
        dx = x - self.x_i
        df_dx = 2 * dx
        df_dla = 0
        return df_dx, df_dla

    def gen(self, t, x, u, ds, psi):
        # extract generalized coordinates and lagrange multipliers
        nq = self.nq
        nla = self.nla
        q = x[:nq]
        la = x[nq:]
        la_arc = x[nq + nla]

        # compute total external force which are scaled by lambda
        f_arc = self.model.forceLawArcLength(1, q, self.u)

        # compute all other force contributions (evaluated at t=1)
        h = self.model.h(1, q, self.u)

        # compute required quantites for the residual
        g = self.model.gap(la_arc, q)
        W_g = self.model.W_g(t, q)

        # build residual
        R = np.zeros(self.nx + 1)
        R[:nq] = h + W_g @ la + la_arc * f_arc
        R[nq:nq+nla] = g
        R[nq + nla] = self.a(la_arc, x, ds, psi) # TODO: la_arc should be in x

        # build residual
        R = np.zeros(self.nx + 1)
        R[:nq] = h + W_g @ la + la_arc * f_arc
        R[nq:nq+nla] = g
        R[nq + nla] = self.a(la_arc, x, ds, psi)
        
        yield R

        # compute required quantities for the tangent matrix
        h_q = self.model.h_q(1, q, self.u)
        g_q = self.model.gap_q(la_arc, q)
        df_dx, df_dla = self.a_x(la_arc, x, ds, psi)
        
    def __residual(self, t, x, u, ds, psi):
        # extract generalized coordinates and lagrange multipliers
        nq = self.nq
        nla = self.nla
        q = x[:nq]
        la = x[nq:]
        
        # note: t in [0, 1] is the force increment used in the arc length method
        la_arc = t

        # compute total external force which are scaled by lambda
        f_arc = self.model.forceLawArcLength(1, q, u)

        # compute all other force contributions (evaluated at t=1)
        h = self.model.h(1, q, u)

        # compute the gap and their force directions
        g = self.model.gap(la_arc, q)
        g_q = self.model.gap_q(la_arc, q)
        W_g = self.model.W_g(t, q)

        # build residual
        R = np.zeros(self.nx + 1)
        R[:nq] = h + W_g @ la + la_arc * f_arc
        R[nq:nq+nla] = g
        R[nq + nla] = self.a(la_arc, x, ds, psi)
        
        return R
        
    def __jacobian(self, t, x, u, ds, psi):
        # extract generalized coordinates and lagrange multipliers
        nq = self.nq
        nla = self.nla
        
        q = x[:nq]
        la = x[nq:]
        
        # note: t in [0, 1] is the force increment used in the arc length method
        la_arc = t

        # generalized force directions of the constraints
        g_q = self.model.gap_q(la_arc, q)

        # compute  tangent of the external forces (evaluated at t=1)
        fe_q = self.model.forceLaws_q(1, q, u)

        # compute external force which are scaled by lambda
        feArcLength = self.model.forceLawArcLength(1, q, u)

        # compute tangent external force which are scaled by lambda
        feArcLength_q = self.model.forceLawArcLength_q(1, q, u)

        # tangent of the internal forces
        fi_q = self.model.internalForces_q(0, q, u)
        
        # compute arc length equation derivatives
        df_dx, df_dla = self.a_x(la_arc, x, ds, psi)

        if self.sparse:
            # gradient of generalized force directions of the constraints
            # contracted with la array
            la_g_qq = self.model.la_gap_qq(t, q, la)
            dR_q_q = fe_q + la_arc * feArcLength_q + fi_q + la_g_qq
            
            # first nq rows of the jacobian
            dR_1 = hstack([dR_q_q, g_q.T, feArcLength[:, None]], format="coo")

            # nla rows of the jacobian
            g_q.resize((nla, nq + nla + 1)) # resizes g_q
            
            # last row for arc length equation
            dR_2 = hstack([df_dx, df_dla], format="coo")

            # put all rows together in a csr matrix
            return vstack([dR_1, g_q, dR_2], format="csr")
        else:
            # gradient of generalized force directions of the constraints
            g_qq = self.model.gap_qq(la_arc, q)
            
            dR = np.zeros((self.nx + 1, self.nx + 1))

            # first nq rows of the jacobian
            dR[:nq, :nq] = fe_q + la_arc * feArcLength_q + fi_q + np.tensordot(g_qq, la, (0, 0))
            dR[:nq, nq:nq+nla] = g_q.T
            dR[:nq, nq + nla] = feArcLength

            # nla rows of the jacobian
            dR[nq:nq+nla, :nq] = g_q

            # last row for arc length equation
            dR[nq + nla, :(nq + nla)] = df_dx
            dR[nq + nla, nq + nla] = df_dla

            return dR
    
    def __init__(self, model, ds, psi=0.0, method='simple', tol=1e-10, maxIter=50, iterGoal=5, laArcInit=1.0e-3, laArcRange=np.array([-1.0, 1.0]), scaleMethod='None'):
        self.maxIter = maxIter
        self.tol = tol
        self.logger = logging.getLogger('Solver')
        self.model = model
        self.ds = ds
        self.psi = psi
        self.iterGoal = iterGoal
        self.laArcInit = laArcInit
        self.laArcRange = laArcRange

        # TODO: think of chosing an initial value for ds, see Crisfield1991 end of section 9.5.1

        if model.__class__.__name__ == 'Model':
            self.sparse = False
            self.linearSolver = np.linalg.solve

        elif model.__class__.__name__ == 'SparseModel':
            self.sparse = True
            self.linearSolver = scipy_spsolve
        else:
            raise TypeError('Wrong model type given. Supported types are Model and SparseModel')

        # dimensions
        self.nq = self.model.assembler.n_qDOF
        self.nla = self.model.assembler.n_laDOF
        # self.nx = self.nq + self.nla
        self.nx = self.nq + self.nla + 1

        # initial conditions
        self.la_arc = [0.0]
        self.q = [self.model.assembler.q0]
        self.la = [self.model.assembler.la0]
        self.u = np.zeros(model.nu) # statics

        # Store converged values of the last force increment here, beginn with initial values.
        # These are necessary for the secant predictor!
        self.x_i = np.concatenate((self.q[0], self.la[0]))
        self.la_arc_i = 0

        # chose constraint equation
        if method == 'simple':
            self.a = self.__f_simple
            self.a_x = self.__df_simple
        elif method == 'Wriggers':
            self.a = self.__f_Wriggers
            self.a_x = self.__df_Wriggers
        elif method == 'Crisfield':
            self.a = self.__f_Crisfield
            self.a_x = self.__df_Crisfield
        else:
            self.a = self.__f_simple
            self.a_x = self.__df_simple
            print('Method "{method}" is not implemented, we use "simple" method insead!')

        # chose scaling strategy
        # TODO: compute cubic polynomial which has horizontal tanget at x=1!
        self.scaleDs = True
        if scaleMethod == 'linear':
            self.__scaleMethod = lambda frac: frac
        elif scaleMethod == 'sqrt':
            self.__scaleMethod = lambda frac: np.sqrt(frac)
        elif scaleMethod == 'x3':
            self.__scaleMethod = lambda frac: (frac - 1.0)**3 + 1.0
        elif scaleMethod == 'tan':
            self.__scaleMethod = lambda frac: np.tan(frac)
        elif scaleMethod == 'exp':
            self.__scaleMethod = lambda frac: np.exp(frac - 1.0)
        else:
            self.__scaleMethod = lambda frac: 0
            self.scaleDs = False
 
    def solve(self):
        # count number of force increments to get first increment with tangential predictor
        forceCounter = 0

        # extract number of generalized coordinates and number of Lagrange multipliers
        nq = self.nq
        nla = self.nla
        nx = nq + nla
        
        # initialize old soltuion values with zeros
        x_old = np.zeros(nx)
        la_arc_old = 0

        # inital values for generalized coordinates, Lagrange multipliers and force scaling
        x = np.concatenate((self.q[0], self.la[0]))
        q = x[:nq]
        u = np.zeros_like(q)
        la_arc = self.laArcInit
        
        # compute initial residual with its euclidean norm
        R = self.__residual(la_arc, x, u, self.ds, self.psi)
        error = np.linalg.norm(R)

        # loop over ranges of lambda
        while la_arc > self.laArcRange[0] and la_arc < self.laArcRange[1]:
            forceCounter += 1

            # use secant predictor for all other force increments than the first one
            # TODO: - see Crisfield1991 section 9.4.3 (9.38) and (9.39) for another possible predictor 
            #         involving the computation of the definiteness of the stiffness matrix of the static
            #         equilibrium (is this related to the current stiffness parameter Philipp has found?)
            #       - oszillation will occur when a bifurcation poin tis found; Crisfield proposes if 
            #         solely the unstable post-buckling path has to be followed, one may instead 
            #         deltaq @ delta p becomes positive (read literature about that!) 
            #         -> this is essentially the current stiffness parameter, see Crisfield1991 section 9.5.2.
            #        - automaticall switch to the (more) stable post-buckling path: see Crisfield1991 
            #          section 9.10 and volume 2!
            #        - restarts, Crisfield1991 section 9.5.5
            #        - predictor solution of higher order, Crisfield1991 section 9.6
            #        - current stiffness parameter: Crisfield1991, section 9.5.2: see also Bergan papers
            #        - A range of alternative or supplementary 'path-measuring parameters' has been
            #          advocated by Eriksson [E3].
            if forceCounter > 1:
                dx = self.x_i - x_old
                dla_arc = self.la_arc_i - la_arc_old
                x += dx
                la_arc += dla_arc
            else:
                dR = self.__jacobian(la_arc, x, u, self.ds, self.psi)
                dx = self.linearSolver(-dR[:nx, :nx], R[:nx])
                x += dx                
            
            # check error of the residual after the predictor step
            R = self.__residual(la_arc, x, u, self.ds, self.psi)
            error = np.linalg.norm(R)

            iter = 0
            self.logger.debug(f'   * iter = {iter}, error = {error:2.4e}')
            while (error > self.tol) and (iter <= self.maxIter):

                # compute jacobian
                dR = self.__jacobian(la_arc, x, u, self.ds, self.psi)
                
                # solve linear system of equation
                # spsolve requires dR to be CSC or CSR matrix format for efficiency!
                # TODO: proof this!
                try:
                    delta_z = self.linearSolver(-dR, R)
                except Exception as e:
                    self.logger.critical(f'Linalg error: {e}')
                    iter = self.maxIter
                    break

                # update x = (q^T, la^T) and la of arc-length method
                x += delta_z[:nx]
                la_arc += delta_z[nx]
                
                # increase Newton counter
                iter += 1
                
                # check for convergence
                R = self.__residual(la_arc, x, u, self.ds, self.psi)
                error = np.linalg.norm(R)
                self.logger.debug(f'   * iter = {iter}, error = {error:2.4e}')

                # Half the arc-length parameter if Newton method does not converge.
                # Reset Newton counter and try again.
                # TODO: We have to limit the number of tries. Otherwise the solver
                #       will never finish if no convergence is achieved!
                if (iter >= self.maxIter) and self.scaleDs:
                    print(f'   * scaled arc-length parameter by 0.5')
                    self.ds *= 0.5
                    iter = 0
                    break

                    # # TODO: we have to restart with smaller ds if no convergence can be achieved after slef.maxIter iterations!
                    # # x = np.copy(self.x_i)
                    # # la_arc = self.la_arc_i
                    # x = np.copy(x_old)
                    # la_arc = la_arc_old
                    
                    # # R = self.__residual(la_arc, x, u, self.ds, self.psi)
                    # # error = np.linalg.norm(R)
                    # dR = self.__jacobian(la_arc, x, u, self.ds, self.psi)
                    # break

            if iter >= self.maxIter:
                self.logger.error(f' - internal Netwon not converged for lambda = {la_arc:2.4e} with error {error:2.2e}.')
                self.logger.error(f'error = {np.max(np.absolute(R))}')
                break
            else:
                self.logger.info(f' - internal Netwon converged for lambda = {la_arc:2.4e} with error {error:2.2e} in {iter} steps.')

            # scale ds such that iter goal is satisfied
            # see Crisfield1991, section 9.5 (9.40) or (9.41) for the quadratic scaling case
            if self.scaleDs:
                # TODO: clean this up
                if iter != 0: # disable scaling if we have halfed the ds parameter before
                    frac = (self.iterGoal) / (iter)
                    self.ds = self.__scaleMethod(frac) * self.ds

                # TODO: as done in the varible time step solvers we should have a minimum and maximum number of increments
                #       and maybe also introduce a safety factor for the increasing step length
                # frac = (self.iterGoal) / (iter)
                # # frac = 1 / frac
                # self.ds = self.__scaleMethod(frac) * self.ds

            # store old equilibrium for secant predictor; store new solution as last converged one
            x_old = np.copy(self.x_i)
            self.x_i = np.copy(x)
            la_arc_old = self.la_arc_i
            self.la_arc_i = la_arc
            
            # append solutions to lists
            self.q.append(np.copy(x[:nq]))
            self.la.append(np.copy(x[nq:]))
            self.la_arc.append(la_arc)

        # return solution object
        return Solution(t=self.la_arc, q=self.q, la=self.la)