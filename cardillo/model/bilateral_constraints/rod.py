import numpy as np
from math import atan2

class Rod():
    r""" Rod between two bodies.

    Parameters
    ----------
    body1 : :mod:`bodies<cardillo.model.bodies>`
        body 1

    body2 : :mod:`bodies<cardillo.model.bodies>`
        body 2
    
    pointIdentifierBody1 : numpy.ndarray, int
        coordinates of point on body 1

    pointIdentifierBody2 : numpy.ndarray, int
        coordinates of point on body 2

    dist : float, :ref:`lambda<python:lambda>`
        distance between body points

    la0 : numpy.ndarray, shape (1,)
        constraint forces at initial configuration

    Returns
    -------
    rod : :class:`RodBodyBody`
        bilateral constraint object

    Notes
    -----
    A rod constraint guarantees a constant distance $L$ between the body points $\\vr_{OP_1}(t,\\vq_1)$ and 
    $\\vr_{OP_2}(t,\\vq_2)$. The Generalized coordinates relevant for this constraint are $\\vq = (\\vq_1,\\vq_2) 
    \\in \\mathbb{R}^f$ given by the generalized coordinates of body 1, $\\vq_1$, and body 2, $\\vq_2$.

    .. figure:: ../img/bilateralConstraints/PivotBodyBody.png
        :figwidth: 50 %
        :align: center
    
    $\\vC_1$ Connectivity matrix of body 1: $\\vq_1 = \\vC_1 \\vq$

    $\\vC_2$ Connectivity matrix of body 2: $\\vq_2 = \\vC_2 \\vq$

    $\\vJ_{P_1}$ Translational Jacobian of body 1 w.r.t. $P_1$: ${}_{I}\\vJ_{P_1} = \\pd{{}_{I}\\vr_{OP_1}}{\\vq_1}$

    $\\vJ_{P_2}$ Translational Jacobian of body 2 w.r.t. $P_2$: ${}_{I}\\vJ_{P_2} = \\pd{{}_{I}\\vr_{OP_2}}{\\vq_2}$

    ``pointIdentifierBody`` has to be choosen according to body specific conventions
    
    - rigid body (${}_K \\vr_{SQ}$): coordinates of vector between center of mass $S$ and body point $Q$ w.r.t. body fixed frame $K$.
    - beam ($\\xi \\in \\{0, 1\\}$): material coordinates of start or end point of beam.

    
    """

    def __init__(self, subsystem1, point_ID1, subsystem2, point_ID2, dist, la_g0=np.zeros(1)):
        self.nla_g = 1

        self.subsystem1 = subsystem1
        self.point_ID1 = point_ID1
        self.r_OP1 = lambda t, q: subsystem1.r_OP(t, q, point_ID1)
        self.r_OP1_q = lambda t, q: subsystem1.r_OP_q(t, q, point_ID1)
        self.J_P1 = lambda t, q: subsystem1.J_P(t, q, point_ID1)
        self.J_P1_q = lambda t, q: subsystem1.J_P_q(t, q, point_ID1)

        self.subsystem2 = subsystem2
        self.point_ID2 = point_ID2
        self.r_OP2 = lambda t, q: subsystem2.r_OP(t, q, point_ID2)
        self.r_OP2_q = lambda t, q: subsystem2.r_OP_q(t, q, point_ID2)
        self.J_P2 = lambda t, q: subsystem2.J_P(t, q, point_ID2)
        self.J_P2_q = lambda t, q: subsystem2.J_P_q(t, q, point_ID2)
        
        self.dist = dist
        self.la_g0 = la_g0

    def assembler_callback(self):
        self.qDOF1 = self.subsystem1.qDOF_P(self.point_ID1)
        self.qDOF2 = self.subsystem2.qDOF_P(self.point_ID2)
        self.qDOF = np.concatenate([self.qDOF1, self.qDOF2])
        self.nq1 = len(self.qDOF1)
        self.nq2 = len(self.qDOF2)
        self.nq = self.nq1 + self.nq2
        
        self.uDOF1 = self.subsystem1.uDOF_P(self.point_ID1)
        self.uDOF2 = self.subsystem2.uDOF_P(self.point_ID2)
        self.uDOF = np.concatenate([self.uDOF1, self.uDOF2])
        self.nu1 = len(self.uDOF1)
        self.nu2 = len(self.uDOF2)
        self.nu = self.nu1 + self.nu2
        
    def g(self, t, q):
        r"""Constraint function.
        .. math::
            \vg = 
            ({}_I \vr_{OP_1}(t, \vq_1) - 
            {}_I \vr_{OP_2}(t, \vq_2))\T ({}_I \vr_{OP_1}(t, \vq_1) - 
            {}_I \vr_{OP_2}(t, \vq_2)) - L^2 \in \mathbb{R} \; .
        """
        nq1 = self.nq1
        r_OP1 = self.r_OP1(t, q[:nq1]) 
        r_OP2 = self.r_OP2(t, q[nq1:])
        return (r_OP1 - r_OP2) @ (r_OP1 - r_OP2)  - self.dist ** 2

    def g_q_dense(self, t, q):
        r"""Partial derivatives of constraint functions. Derivatives w.r.t. generalized coordinates.

        .. math::
            \pd{\vg}{\vq} & =
            2 ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2))\T 
            \left(\pd{{}_{I}\vr_{OP_1}}{\vq_1}(t, \vq_1) \pd{\vq_1}{\vq} - 
            \pd{{}_{I}\vr_{OP_2}}{\vq_2}(t, \vq_2) \pd{\vq_2}{\vq} \right)\\ 
            & = 2 ({}_I \vr_{OP_1}(t, \vq_1) - 
            {}_I \vr_{OP_2}(t, \vq_2))\T \left({}_{I}\vJ_{P_1}(t, \vq_1) \vC_1 - 
            {}_{I}\vJ_{P_2}(t, \vq_2) \vC_2\right) \\ & =
            2 ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2))\T
            \Big({}_{I}\vJ_{P_1}(t, \vq_1), -{}_{I}\vJ_{P_2}(t, \vq_2)\Big)
            \in \mathbb{R}^{1 \times f} \; .

        The term $\\pd{g_i}{q^j}$ is stored in ``g_q[i, j]``.
        """
        nq1 = self.nq1
        r_OP1 = self.r_OP1(t, q[:nq1]) 
        r_OP2 = self.r_OP2(t, q[nq1:])
        r_OP1_q = self.r_OP1_q(t, q[:nq1]) 
        r_OP2_q = self.r_OP2_q(t, q[nq1:])
        return np.array([2 * (r_OP1 - r_OP2) @ np.hstack([r_OP1_q,-r_OP2_q])])

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))
   
    def W_g_dense(self, t, q):
        nq1 = self.nq1
        r_P2P1 = self.r_OP1(t, q[:nq1]) - self.r_OP2(t, q[nq1:])
        J_P1 = self.J_P1(t, q[:nq1]) 
        J_P2 = self.J_P2(t, q[nq1:])
        return 2 * np.array([ np.concatenate([J_P1.T @ r_P2P1, -J_P2.T @ r_P2P1])]).T

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nq1 = self.nq1
        nu1 = self.nu1
        r_P2P1 = self.r_OP1(t, q[:nq1]) - self.r_OP2(t, q[nq1:])
        r_OP1_q = self.r_OP1_q(t, q[:nq1]) 
        r_OP2_q = self.r_OP2_q(t, q[nq1:])
        J_P1 = self.J_P1(t, q[:nq1]) 
        J_P2 = self.J_P2(t, q[nq1:])
        J_P1_q = self.J_P1_q(t, q[:nq1]) 
        J_P2_q = self.J_P2_q(t, q[nq1:])

        # dense blocks
        dense = np.zeros((self.nu, self.nq))

        dense[:nu1, :nq1] = J_P1.T @ r_OP1_q + np.einsum('i,ijk->jk',r_P2P1, J_P1_q)
        dense[:nu1, nq1:] = - J_P1.T @ r_OP2_q
        dense[nu1:, :nq1] = - J_P2.T @ r_OP1_q
        dense[nu1:, nq1:] = J_P2.T @ r_OP2_q - np.einsum('i,ijk->jk',r_P2P1, J_P2_q)

        coo.extend(2 * la_g[0] * dense, (self.uDOF, self.qDOF))