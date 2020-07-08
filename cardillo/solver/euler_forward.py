import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import csr_matrix

from cardillo.solver import Solution

class Euler_forward():
    def __init__(self, model, t1, dt):
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)
        
        # constant time step
        self.dt = dt

    def step(self, tk, qk, uk):
        # general quantities
        dt = self.dt

        tk1 = tk + dt
        uk1 = uk + dt * spsolve(self.model.M(tk, qk, scipy_matrix=csr_matrix), self.model.h(tk, qk, uk))
        qk1 = qk + dt * self.model.q_dot(tk, qk, uk)
        
        return tk1, qk1, uk1

    def solve(self): 
        
        # lists storing output variables
        tk = self.model.t0
        qk = self.model.q0.copy()
        uk = self.model.u0.copy()
        
        q = [qk]
        u = [uk]

        for tk in self.t[:-1]:
            tk1, qk1, uk1 = self.step(tk, qk, uk)

            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            # update local variables for accepted time step
            tk, qk, uk = tk1, qk1, uk1
            
        # write solutio
        return Solution(t=self.t, q=np.array(q), u=np.array(u))