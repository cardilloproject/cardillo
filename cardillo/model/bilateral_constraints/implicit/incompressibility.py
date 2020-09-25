import numpy as np
from cardillo.math.algebra import determinant, inverse

# TODO: F_qp nur einmal berechnen
class Incompressibility():
    def __init__(self, subsystem, la_mesh):
        self.subsystem = subsystem
        self.mesh = subsystem.mesh
        self.la_mesh = la_mesh
        self.nla_g = la_mesh.nn

        self.qDOF = self.subsystem.qDOF
        #self.uDOF = self.subsystem.uDOF

    # assembler callback??

    def g_el(self, t, qe, el):
        ge = np.zeros(self.la_mesh.nn_el)
        for i in range(self.mesh.nqp):
            F_qp = np.zeros((self.subsystem.dim, self.subsystem.dim))
            for a in range(self.mesh.nn_el):
                qa = qe[self.mesh.nodalDOF[a]]
                F_qp += np.outer(qa, self.subsystem.N_X[el, i, a])

            detF = determinant(F_qp)
            for a_tilde in range(self.la_mesh.n):
                ge[a_tilde] += self.la_mesh.N[el, i, a_tilde]  * (detF - 1) * self.subsystem.w_J0[el, i]
       
        return ge


    def g(self, t, q):
        g = np.zeros(self.nla_q)
        for el in range(self.mesh.nel):
            elDOF_el = self.mesh.elDOF[el]
            qe = q[elDOF_el]            
            la_elDOF_el = self.la_mesh.elDOF[el]  
            g[la_elDOF_el] += self.g_el(t, qe, el)

        return g


    def g_el_q(self, t, qe, el):
        ge_q = np.zeros((self.la_mesh.nn_el, qe.shape[0]))
        F_qp = np.zeros((self.subsystem.dim, self.subsystem.dim))
        for a in range(self.mesh.nn_el):
            qa = qe[self.mesh.nodalDOF[a]]
            F_qp += np.outer(qa, self.subsystem.N_X[el, i, a])

        F_inv = inverse(F_qp)
        detF = determinant(F_qp)
        for a in range(self.mesh.n):
            for a_tilde in range(self.la_mesh.nn_el):
                ge_q[np.ix_(np.array([a_tilde]), self.mesh.nodalDOF[a])] += self.la_mesh.N[el, i, a_tilde] *  detF * np.einsum('kl,l', F_inv.T,  self.body.N_X[el, i, a]) * self.body.w_J0[el, i]
        
        return ge_q


    def g_q_dense(self, t, q):
        """First partial derivative of the constraint function w.r.t. generalized coordinates.
        """
        g_q = np.zeros((self.nla_q, self.subsystem.nz))
        for el in range(self.mesh.nel):
            elDOF_el = self.mesh.elDOF[el]
            qe = q[elDOF_el]
            la_elDOF_el = self.la_mesh.elDOF[el]  
            g_q[np.ix_(la_elDOF_el , elDOF_el)] += self.g_el_q(t, qe, el)
        return g_q

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q).T, self.qDOF, self.la_gDOF)

    def Wla_g_q_el(self, t, qel, lael, el):
        Wla_g_q_el = np.zeros((qel.shape[0], qel.shape[0]))
        for i in range(self.mesh.nqp):
            F_qp = np.zeros((self.subsystem.dim, self.subsystem.dim))
            dF_dq = np.zeros((self.subsystem.dim, self.subsystem.dim, self.mesh.nq_n * self.mesh.nn_el))
            for a in range(self.mesh.nn_el):
                # TODO: reference
                qa = qel[self.mesh.nodalDOF[a]]
                F_qp += np.outer(qa, self.body.N_X[el, i, a])
                dF_dq[:, :, self.mesh.nodalDOF[a]] += np.einsum('ik,j->ijk', np.eye(self.subsystem.dim), self.subsystem.N_X[el, i, a])
            
            F_inv = inverse(F_qp)
            detF = determinant(F_qp)
            dJFinvT_dqe = detF * np.einsum('klmn, mnj->klj', (np.einsum('nm,lk->klmn',bF_inv, F_inv) - np.einsum('lm,nk->klmn', F_inv, F_inv)),  dF_dq) 
            for a in range(self.mesh.nn_el):
                for a_tilde in range(self.la_mesh.nn_el):
                    la_a_tilde = lael[a_tilde]
                    Wla_g_q_el[self.mesh.nodalDOF[a], :] += la_a_tilde * self.la_mesh.N[el, i, a_tilde] *  np.einsum('klj,l->kj', dJFinvT_dqe,  self.subsystem.N_X[el,i,a]) * self.subsystem.w_J0[el,i]
        
        return Wla_g_q_el


    def Wla_g_q(self, t, q, la_g, coo):
        Wla_g_q = np.zeros((self.subsystem.nz, self.subsystem.nz))
        for el in range(self.mesh.nel):
            qel = q[self.mesh.elDOF[el]]
            la_elDOF = self.la_mesh.elDOF[el]  
            lael = la_g[la_elDOF]
            Wla_g_q[np.ix_(self.mesh.elDOF[el], self.mesh.elDOF[el])] += self.Wla_g_q_el(t, qel, lael, el)

        coo.extend(Wla_g_q, (self.qDOF, self.qDOF))