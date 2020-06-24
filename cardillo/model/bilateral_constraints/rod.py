import numpy as np
from math import atan2

# from cardillo.utility.transformations import rot

# class RodBodyEnv(BilateralConstraint):
#     r""" Rod between body and envionment.

#     Parameters
#     ----------
#     body : :mod:`bodies<cardillo.model.bodies>`
#         body
    
#     point_ID_Body : numpy.ndarray, int
#         coordinates of body point

#     dist : float, :ref:`lambda<python:lambda>`
#         distance between body points
            
#     rPivot : numpy.ndarray, :ref:`lambda<python:lambda>`
#         coordinates of spatial point w.r.t. inertial frame

#     vPivot : :ref:`lambda<python:lambda>`
#         velocity of spatial point w.r.t. inertial frame

#     aPivot : :ref:`lambda<python:lambda>`
#         acceleration of spatial point w.r.t. inertial frame

#     la0 : numpy.ndarray, shape (1,)
#         constraint forces at initial configuration

#     Returns
#     -------
#     rod : :class:`RodBodyEnv`
#         bilateral constraint object

#     Notes
#     -----
#     A rod constraint guarantees a constant distance $L$ between the body point 
#     $\\vr_{OQ}(t,\\vq)$ and the spatial point $\\vr_{OP}(t)$.

#     .. figure:: ../img/bilateralConstraints/PivotBodyEnv.png
#         :figwidth: 50 %
#         :align: center

#     $\\vJ_{Q}$ Translational Jacobian of body w.r.t. $Q$: ${}_{I}\\vJ_{Q} = \\pd{{}_{I}\\vr_{OQ}}{\\vq}$

#     ``point_ID_Body`` has to be choosen according to body specific conventions
    
#     - rigid body (${}_K \\vr_{SQ}$): coordinates of vector between center of mass $S$ and body point $Q$ w.r.t. body fixed frame $K$.
#     - beam ($\\xi \\in \\{0, 1\\}$): material coordinates of start or end point of beam.

#     """

#     def __init__(self, body, point_ID_Body, dist=0, rPivot=None, vPivot=None, aPivot=None, la0=np.zeros(1)):

#         self.n_laDOF = 1                            # dimension of constraint forces
#         self.body = body                            
#         self.point_ID_ = point_ID_Body
#         self.dist = dist
#         self.la0 = la0

#         if rPivot is None:
#             try:
#                 self.rPivot = lambda t: body.position(t, body.Q, point_ID_Body)
#                 self.vPivot = lambda t: 0 * body.position(t, body.Q, point_ID_Body)
#                 self.aPivot = lambda t: 0 * body.position(t, body.Q, point_ID_Body)
#             except:
#                 print('JunctionBodyEnv: point or reference config. must be specified!')

#         elif not callable(rPivot):
#             self.rPivot = lambda t: rPivot
#             self.vPivot = lambda t: 0 * rPivot
#             self.aPivot = lambda t: 0 * rPivot
#         else:
#             self.rPivot = rPivot
#             if vPivot is not None:
#                 self.vPivot = vPivot
#             else:
#                 eps = 1e-6
#                 self.vPivot = lambda t: (rPivot(t+eps) - rPivot(t-eps)) / (2 * eps)
            
#             if aPivot is not None:
#                 self.aPivot = aPivot
#             else:
#                 eps = 1e-6
#                 self.aPivot = lambda t: (self.vPivot(t+eps) - self.vPivot(t-eps)) / (2 * eps)


#     def get_qDOF(self):
#         r"""Indices of generalized coordinates for involved bodies.

#         Returns
#         -------
#         qDOF : numpy.ndarray, shape (f,)
#             indices of generalized coordinates ``[body.qDOF]``
#         """
#         # returns degrees of freedom of involved bodies for assembler
#         return self.body.qDOF

#     def gap(self, t, q):
#         r"""Gap functions.

#         Parameters
#         ----------
#         t : float
#             time instant

#         q : numpy.ndarray, shape (f,)
#             generalized coordinates of involved bodies

#         Returns
#         -------
#         g : numpy.ndarray, shape (1,)
#             gap functions

#         Notes
#         -----
#         Gap functions 
        
#         .. math::
#             \vg = 
#             ({}_I \vr_{OQ}(t, \vq) - {}_I \vr_{OP}(t))\T 
#             ({}_I \vr_{OQ}(t, \vq) - {}_I \vr_{OP}(t)) - L^2 \in \mathbb{R} \; .
#         """
#         r_OQ = self.body.position(t, q, self.point_ID_)
#         return (r_OQ - self.rPivot(t)) @ (r_OQ - self.rPivot(t))  - self.dist ** 2

#     def gap_t(self, t, q):
#         r"""Partial time derivative of gap functions.
        
#         Parameters
#         ----------
#         t : float
#             time instant

#         q : numpy.ndarray, shape (f,)
#             generalized coordinates of involved bodies

#         Returns
#         -------
#         g_t : numpy.ndarray, shape (1,)
#             partial time derivative of gap functions
        
#         Notes
#         -----
#         Partial time derivative of gap functions

#         .. math::
#             \pd{\vg}{t} = 
#             2 \left(\pd{{}_I \vr_{OQ}}{t}(t, \vq) - \pd{{}_I \vr_{OP}}{t}(t)\right)^\mathrm{T}
#             ({}_I \vr_{OQ}(t, \vq) - {}_I \vr_{OP}(t)) \in \mathbb{R} \; .
#         """

#         r_OQ = self.body.position(t, q, self.point_ID_)
#         r_OQ_t = self.body.position_t(t, q, self.point_ID_)

#         return 2 * (r_OQ_t - self.vPivot(t)) @ (r_OQ - self.rPivot(t))


#     def gap_tt(self, t, q):
#         r"""Second partial time derivative of gap functions.

#         Parameters
#         ----------
#         t : float
#             time instant

#         q : numpy.ndarray, shape (f,)
#             generalized coordinates of involved bodies

#         Returns
#         -------
#         g_tt : numpy.ndarray, shape (1,)
#             second partial time derivative of gap functions
        
#         Notes
#         -----
#         Second partial time derivative of gap functions

#         .. math::
#             \frac{\partial^2 \vg}{\partial t^2} = & 
#             2 \left( \frac{\partial^2 {}_I \vr_{OQ}}{\partial t^2}(t, \vq) - 
#             \frac{\partial^2 {}_I \vr_{OP}}{\partial t^2}(t) \right)^\mathrm{T} ({}_I \vr_{OQ}(t, \vq) - {}_I \vr_{OP}(t)) \\
#             &+ 2 \left(\pd{{}_I \vr_{OQ}}{t}(t, \vq) - 
#             \pd{{}_I \vr_{OP}}{t}(\vq)\right)^\mathrm{T} \left(\pd{{}_I \vr_{OQ}}{t}(t, \vq) - 
#             \pd{{}_I \vr_{OP}}{t}(t)\right) \in \mathbb{R} \; .

#         """
                
#         r_OQ = self.body.position(t, q, self.point_ID)
#         r_OQ_t = self.body.position_t(t, q, self.point_ID)
#         r_OQ_tt = self.body.position_tt(t, q, self.point_ID)
        
#         return 2 * (r_OQ_tt - self.aPivot(t)) @  (r_OQ - self.rPivot(t)) + 2 * (r_OQ_t - self.vPivot(t)) @ (r_OQ_t - self.vPivot(t))

#     def gap_q(self, t, q):
#         r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates.

#         Parameters
#         ----------
#         t : float
#             time instant

#         q : numpy.ndarray, shape (f,)
#             generalized coordinates of involved bodies

#         Returns
#         -------
#         g_q : numpy.ndarray, shape (1, f)
#             partial derivatives of gap functions w.r.t. generalized coordinates

#         Notes
#         -----
#         Partial derivatives of gap functions w.r.t. generalized coordinates

#         .. math::
#             \pd{\vg}{\vq} & =
#             2 ({}_I \vr_{OQ}(t, \vq) - {}_I \vr_{OP}(t))\T \pd{{}_{I}\vr_{OQ}}{\vq}(t, \vq)\\ 
#             & = 2 ({}_I \vr_{OQ}(t, \vq) - 
#             {}_I \vr_{OP}(t))\T {}_{I}\vJ_{Q}(t, \vq)
#             \in \mathbb{R}^{1 \times f} \; .

#         The term $\\pd{g_i}{q^j}$ is stored in ``g_q[i, j]``.
#         """
#         r_OQ = self.body.position(t, q, self.point_ID) 
#         JQ = self.body.position_q(t, q, self.point_ID)
        
#         return 2 * (r_OQ - self.rPivot(t)) @ JQ

#     def gap_qt(self, t, q):
#         r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates and time.

#         Parameters
#         ----------
#         t : float
#             time instant

#         q : numpy.ndarray, shape (f,)
#             generalized coordinates of involved bodies

#         Returns
#         -------
#         g_qt : numpy.ndarray, shape (1, f)
#             partial derivatives of gap functions w.r.t generalized coordinates and time

#         Notes
#         -----
#         Partial derivatives of gap functions w.r.t. generalized coordinates and time

#         .. math::
#             \frac{\partial^2 \vg}{\partial \vq \partial t} & = 
#             2 \left(\pd{{}_I \vr_{OQ}}{t}(t, \vq) - 
#             \pd{{}_I \vr_{OP}}{t}(t)\right)^\mathrm{T}
#             {}_{I}\vJ_{Q}(t, \vq) \\ 
#             & + 
#             2 ({}_I \vr_{OQ}(t, \vq) - {}_I \vr_{OP}(t))\T
#             \pd{{}_{I}\vJ_{Q}}{t}(t, \vq) \in \mathbb{R}^{1 \times f} \; .

#         The term $\\frac{\\partial^2 g_i}{\\partial q^j \\partial t}$ is stored in ``g_qt[i, j]``.
#         """
#         r_OQ = self.body.position(t, q, self.point_ID)
#         r_OQ_t = self.body.position_t(t, q, self.point_ID)
#         JQ = self.body.position_q(t, q, self.point_ID)
#         JQ_t = self.body.position_qt(t, q, self.point_ID) 

#         return 2 * JQ.T @ (r_OQ_t - self.vPivot(t)) + 2 * (r_OQ - self.rPivot(t)) @ JQ_t


#     def gap_qtt(self, t, q):
#         r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates and time

#         Parameters
#         ----------
#         t : float
#             time instant

#         q : numpy.ndarray, shape (f,)
#             generalized coordinates of involved bodies

#         Returns
#         -------
#         g_qtt : numpy.ndarray, shape (1, f)
#             partial derivatives of gap functions w.r.t. generalized coordinates and time

#         Notes
#         -----
#         Partial derivatives of gap functions w.r.t. generalized coordinates and time

#         .. math::
#             \frac{\partial^3 \vg}{\partial \vq \partial t^2} & =
#             2 \left( \frac{\partial^2 {}_I \vr_{OQ}}{\partial t^2}(t, \vq) - 
#             \frac{\partial^2 {}_I \vr_{OP}}{\partial t^2}(t) \right)^\mathrm{T}
#             {}_{I}\vJ_{Q}(t, \vq) \\
#             & + 4 \left(\pd{{}_I \vr_{OQ}}{t}(t, \vq) - \pd{{}_I \vr_{OP}}{t}(t)\right)^\mathrm{T}
#             \pd{{}_{I}\vJ_{Q}}{t}(t, \vq) \\
#             & + 2 ({}_I \vr_{OQ}(t, \vq) - {}_I \vr_{OP}(t))\T
#             \frac{\partial^2 {}_{I}\vJ_Q}{\partial t^2}(t, \vq) \in \mathbb{R}^{1 \times f} \; .

#         The term $\\frac{\\partial^3 g_i}{\\partial q^j \\partial t^2}$ is stored in ``g_qtt[i, j]``.
#         """

#         r_OQ = self.body.position(t, q, self.point_ID)
#         r_OQ_t = self.body.position_t(t, q, self.point_ID)
#         r_OQ_tt = self.body.position_tt(t, q, self.point_ID)
#         JQ = self.body.position_q(t, q, self.point_ID)
#         JQ_t = self.body.position_qt(t, q, self.point_ID) 
#         JQ_tt = self.body.position_qtt(t, q, self.point_ID) 

#         tmp1 = 2 * (r_OQ_tt - self.aPivot(t)) @ JQ
#         tmp2 = 4 * (r_OQ_t  - self.vPivot(t)) @ JQ
#         tmp3 = 2 * (r_OQ    - self.rPivot(t)) @ JQ
        
#         return tmp1 + tmp2 + tmp3

#     def gap_qq(self, t, q):
#         r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates.

#         Parameters
#         ----------
#         t : float
#             time instant

#         q : numpy.ndarray, shape (f,)
#             generalized coordinates of involved bodies

#         Returns
#         -------
#         g_qq : numpy.ndarray, shape (1, f, f)
#             second partial derivative of gap functions w.r.t. generalized coordinates

#         Notes
#         -----
#         Second partial derivatives of gap functions w.r.t. generalized coordinates

#          .. math::
#             \frac{\partial^2 \vg}{\partial \vq^2} = 
#             2 \Big({}_I\vJ_{Q}(t, \vq)\T{}_I\vJ_{Q}(t, \vq) + 
#             ({}_I \vr_{OQ}(t, \vq) - 
#             {}_I \vr_{OP}(t)) \cdot \pd{{}_{I} \vJ_{OQ}}{\vq}(t, \vq)\Big)
#             \in \mathbb{R}^{1 \times f \times f} \; .


#         The term $\\frac{\\partial^2 g_i}{\\partial q^j \\partial q^k}$ is stored in ``g_qq[i, j, k]``.
#         """
        
#         r_OQ = self.body.position(t, q, self.point_ID)
#         JQ = self.body.position_q(t, q, self.point_ID)
#         JQ_q = self.body.position_qq(t, q, self.point_ID) 

#         return 2 * (JQ.T @ JQ + np.tensordot((r_OQ - self.rPivot(t)), JQ_q, 1)) # 2 * (JQ.T @ JQ + (JQ_q.T @ (r_OQ - self.rPivot(t))).T )


#     def gap_qqt(self, t, q):
#         r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates and time.

#         Parameters
#         ----------
#         t : float
#             time instant

#         q : numpy.ndarray, shape (f,)
#             generalized coordinates of involved bodies

#         Returns
#         -------
#         g_qqt : numpy.ndarray, shape (1, f, f)
#             partial derivatives of gap functions w.r.t. generalized coordinates and time

#         Notes
#         -----
#         Partial derivatives of gap functions w.r.t. generalized coordinates and time

#         .. math::
#             \frac{\partial^3 \vg}{\partial \vq^2 \partial t} = 
#             &2  \Big(\pd{{}_I\vJ_{Q}}{t}(t, \vq)\T{}_I\vJ_{Q}(t, \vq) + 
#             {}_I\vJ_{Q}(t, \vq)\T\pd{{}_I\vJ_{Q}}{t}(t, \vq) \\ & +
#             \Big(\pd{{}_I \vr_{OQ}}{t}(t, \vq) - 
#             \pd{{}_I \vr_{OP}}{t}(t)\Big) \cdot \pd{{}_{I} \vJ_{OQ}}{\vq}(t, \vq) \\
#             & + ({}_I \vr_{OQ}(t, \vq) - {}_I \vr_{OP}(t)) \cdot 
#             \frac{\partial^2 {}_{I} \vJ_{OQ}}{\partial \vq \partial t}(t, \vq)\Big) 
#             \in \mathbb{R}^{1 \times f \times f} \; .

#         The term $\\frac{\\partial^3 g_i}{\\partial q^j \\partial q^k \\partial t}$ is stored in ``g_qqt[i, j, k]``.
#         """
#         nq = self.body.n_qDOF
#         r_OQ = self.body.position(t, q, self.point_ID)
#         r_OQ_t = self.body.position_t(t, q, self.point_ID)
#         JQ = self.body.position_q(t, q, self.point_ID)
#         JQ_t = self.body.position_qt(t, q, self.point_ID) 
#         JQ_q = self.body.position_qq(t, q, self.point_ID)
#         JQ_qt = self.body.position_qqt(t, q, self.point_ID) 

#         g_qqt = np.zeros((1, nq, nq))

#         tmp1 = JQ_t.T @ JQ + JQ.T @ JQ_t
#         tmp2 = np.tensordot((r_OQ_t - self.vPivot(t)), JQ_q, 1) # tmp2 = (JQ_q.T @ (r_OQ_t - self.vPivot(t))).T
#         tmp3 = np.tensordot((r_OQ - self.rPivot(t)), JQ_qt, 1) # tmp3 = (JQ_qt.T @ (r_OQ - self.rPivot(t))).T
        
#         g_qqt[0, :, :] =  2 * (tmp1 + tmp2 + tmp3)

#         return g_qqt

#     def gap_qqq(self, t, q):
#         r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates.

#         Parameters
#         ----------
#         t : float
#             time instant

#         q : numpy.ndarray, shape (f,)
#             generalized coordinates of involved bodies

#         Returns
#         -------
#         g_qqq : numpy.ndarray, shape (1, f, f, f)
#             triple partial derivative of gap functions w.r.t. generalized coordinates

#         Notes
#         -----        
#         Triple partial derivatives of gap functions w.r.t. generalized coordinates. 
#         We use the following abbreviations for the indices: $({}_I\\vr_{OQ})_i = r^Q_i$, 
#         $({}_I\\vr_{OP})_i = r^P_i$, $({}_I\\vJ_{Q})_{ij} = J_{ij}$.


#         .. math::
#             \frac{\partial^3 g}{\partial q_i \partial q_j \partial q_k} =
#             \sum_{l = 1}^{2} 2 \left[
#             \pd{J_{li}}{q_k} J_{lj} + J_{li} \pd{J_{lj}}{q_k}
#             + J_{lk} \pd{J_{li}}{q_j} + (r^Q_l - r^P_l) \frac{\partial^2 J_{li}}{\partial q_j \partial q_k}\right]
            

#         The term $\\frac{\\partial^3 g_i}{\\partial q^j \\partial q^k \\partial q^l}$ is stored in ``g_qqq[i, j, k, l]``.
#         """

#         nq = self.body.n_qDOF

#         r_OQ = self.body.position(t, q, self.point_ID) 
#         JQ = self.body.position_q(t, q, self.point_ID)
#         JQ_q = self.body.position_qq(t, q, self.point_ID) 
#         JQ_qq = self.body.position_qqq(t, q, self.point_ID) 


#         g_qqq = np.zeros((1, nq, nq, nq))

#         for i in range(nq):
#             for j in range(nq):
#                 for k in range(nq):
#                     for l in range(2):
#                         g_qqq[0, i, j, k] += 2 * (JQ_q[l, i, k] * JQ[l, j] + JQ[l, i] * JQ_q[l, j, k] \
#                                                                 + JQ[l, k] * JQ_q[l, i, j] + (r_OQ[l] - self.rPivot(t)[l]) * JQ_qq[l, i, j, k])   

#         return g_qqq


class RodBodyBody():
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

    def __init__(self, point1, point2, dist, la_g0=np.zeros(1)):
        self.nla_g = 1
        self.point1 = point1
        self.point2 = point2
        self.dist = dist
        self.la_g0 = la_g0

    @property
    def qDOF(self):
        return np.concatenate([self.point1.qDOF, self.point2.qDOF])

    @property
    def uDOF(self):
        return np.concatenate([self.point1.uDOF, self.point2.uDOF])

    def g(self, t, q):
        r"""Constraint function.
        .. math::
            \vg = 
            ({}_I \vr_{OP_1}(t, \vq_1) - 
            {}_I \vr_{OP_2}(t, \vq_2))\T ({}_I \vr_{OP_1}(t, \vq_1) - 
            {}_I \vr_{OP_2}(t, \vq_2)) - L^2 \in \mathbb{R} \; .
        """

        nq1 = len(self.point1.qDOF)
        r_OP1 = self.point1.position(t, q[:nq1]) 
        r_OP2 = self.point2.position(t, q[nq1:])
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
        nq1 = len(self.point1.qDOF)
        r_OP1 = self.point1.position(t, q[:nq1]) 
        r_OP2 = self.point2.position(t, q[nq1:])
        J1 = self.point1.position_q(t, q[:nq1]) 
        J2 = self.point2.position_q(t, q[nq1:])
        return np.array([2 * (r_OP1 - r_OP2) @ np.hstack([J1,-J2])])

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def B_dense(self, t, q):
        nq1 = len(self.point1.qDOF)
        nq2 = len(self.point2.qDOF)
        nu1 = len(self.point1.uDOF)
        nu2 = len(self.point2.uDOF)
        B = np.zeros((nq1 + nq2, nu1 + nu2))
        
        B[:nq1, :nu1] = self.point1.B(t, q[:nq1])
        B[nq1:, nu1:] = self.point2.B(t, q[nq1:])

        return B
    
    def W_g_dense(self, t, q):
        return (self.g_q_dense(t, q) @ self.B_dense(t, q)).T

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g(self, t, q, la_g):
        return self.W_g_dense(t, q) @ la_g

    # def __g_t(self, t, q):
    #     r"""Partial time derivative of gap functions.
        
    #     Parameters
    #     ----------
    #     t : float
    #         time instant

    #     q : numpy.ndarray, shape (f,)
    #         generalized coordinates of involved bodies

    #     Returns
    #     -------
    #     g_t : numpy.ndarray, shape (1,)
    #         partial time derivative of gap functions
        
    #     Notes
    #     -----
    #     Partial time derivative of gap functions

    #     .. math::
    #         \pd{\vg}{t} = 
    #         2 \left(\pd{{}_I \vr_{OP_1}}{t}(t, \vq_1) - 
    #         \pd{{}_I \vr_{OP_2}}{t}(t, \vq_2)\right)^\mathrm{T}
    #         ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2)) \in \mathbb{R} \; .

    #     """
    #     r_OP1 = self.point1.position(t, q[:self.nq1]) 
    #     r_OP2 = self.point1.position(t, q[self.nq1:])
    #     r_OP1_t = self.point1.position_t(t, q[:self.nq1]) 
    #     r_OP2_t = self.point1.position_t(t, q[self.nq1:])
    #     return 2 * (r_OP1_t - r_OP2_t) @ (r_OP1 - r_OP2)

    # def gap_tt(self, t, q):
    #     r"""Second partial time derivative of gap functions.

    #     Parameters
    #     ----------
    #     t : float
    #         time instant

    #     q : numpy.ndarray, shape (f,)
    #         generalized coordinates of involved bodies

    #     Returns
    #     -------
    #     g_tt : numpy.ndarray, shape (1,)
    #         second partial time derivative of gap functions
        
    #     Notes
    #     -----
    #     Second partial time derivative of gap functions

    #     .. math::
    #         \frac{\partial^2 \vg}{\partial t^2} = &
    #         2 \left( \frac{\partial^2 {}_I \vr_{OP_1}}{\partial t^2}(t, \vq_1) - 
    #         \frac{\partial^2 {}_I \vr_{OP_2}}{\partial t^2}(t, \vq_2) \right)^\mathrm{T}
    #         ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2)) \\
    #         &+ 2 \left(\pd{{}_I \vr_{OP_1}}{t}(t, \vq_1) - 
    #         \pd{{}_I \vr_{OP_2}}{t}(t, \vq_2)\right)^\mathrm{T} \left(\pd{{}_I \vr_{OP_1}}{t}(t, \vq_1) - 
    #         \pd{{}_I \vr_{OP_2}}{t}(t, \vq_2)\right) \in \mathbb{R} \; .
    #     """
    #     nq1 = self.body1.n_qDOF
    #     r_OP1 = self.body1.position(t, q[:nq1], self.point_ID_1) 
    #     r_OP2 = self.body2.position(t, q[nq1:], self.point_ID_2)
    #     r_OP1_t = self.body1.position_t(t, q[:nq1], self.point_ID_1) 
    #     r_OP2_t = self.body2.position_t(t, q[nq1:], self.point_ID_2)
    #     r_OP1_tt = self.body1.position_tt(t, q[:nq1], self.point_ID_1) 
    #     r_OP2_tt = self.body2.position_tt(t, q[nq1:], self.point_ID_2)
    #     return 2 * (r_OP1_tt - r_OP2_tt) @  (r_OP1 - r_OP2) + 2 * (r_OP1_t - r_OP2_t) @ (r_OP1_t - r_OP2_t)

    # def gap_q(self, t, q):
    #     r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates.

    #     Parameters
    #     ----------
    #     t : float
    #         time instant

    #     q : numpy.ndarray, shape (f,)
    #         generalized coordinates of involved bodies

    #     Returns
    #     -------
    #     g_q : numpy.ndarray, shape (1, f)
    #         partial derivatives of gap functions w.r.t. generalized coordinates

    #     Notes
    #     -----
    #     Partial derivatives of gap functions w.r.t. generalized coordinates

    #     .. math::
    #         \pd{\vg}{\vq} & =
    #         2 ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2))\T 
    #         \left(\pd{{}_{I}\vr_{OP_1}}{\vq_1}(t, \vq_1) \pd{\vq_1}{\vq} - 
    #         \pd{{}_{I}\vr_{OP_2}}{\vq_2}(t, \vq_2) \pd{\vq_2}{\vq} \right)\\ 
    #         & = 2 ({}_I \vr_{OP_1}(t, \vq_1) - 
    #         {}_I \vr_{OP_2}(t, \vq_2))\T \left({}_{I}\vJ_{P_1}(t, \vq_1) \vC_1 - 
    #         {}_{I}\vJ_{P_2}(t, \vq_2) \vC_2\right) \\ & =
    #         2 ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2))\T
    #         \Big({}_{I}\vJ_{P_1}(t, \vq_1), -{}_{I}\vJ_{P_2}(t, \vq_2)\Big)
    #         \in \mathbb{R}^{1 \times f} \; .

    #     The term $\\pd{g_i}{q^j}$ is stored in ``g_q[i, j]``.
    #     """
    #     nq1 = self.body1.n_qDOF
    #     r_OP1 = self.body1.position(t, q[:nq1], self.point_ID_1) 
    #     r_OP2 = self.body2.position(t, q[nq1:], self.point_ID_2)
    #     J1 = self.body1.position_q(t, q[:nq1], self.point_ID_1) 
    #     J2 = self.body2.position_q(t, q[nq1:], self.point_ID_2)
    #     return 2 * (r_OP1 - r_OP2) @ np.hstack([J1,-J2])

    # def gap_qt(self, t, q):
    #     r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates and time.

    #     Parameters
    #     ----------
    #     t : float
    #         time instant

    #     q : numpy.ndarray, shape (f,)
    #         generalized coordinates of involved bodies

    #     Returns
    #     -------
    #     g_qt : numpy.ndarray, shape (1, f)
    #         partial derivatives of gap functions w.r.t generalized coordinates and time

    #     Notes
    #     -----
    #     Partial derivatives of gap functions w.r.t. generalized coordinates and time

    #     .. math::
    #         \frac{\partial^2 \vg}{\partial \vq \partial t} & = 
    #         2 \left(\pd{{}_I \vr_{OP_1}}{t}(t, \vq_1) - 
    #         \pd{{}_I \vr_{OP_2}}{t}(t, \vq_2)\right)^\mathrm{T}
    #         \left({}_{I}\vJ_{P_1}(t, \vq_1) \vC_1 - 
    #         {}_{I}\vJ_{P_2}(t, \vq_2) \vC_2\right) \\ 
    #         & + 
    #         2 ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2))\T
    #         \left(\pd{{}_{I}\vJ_{P_1}}{t}(t, \vq_1) \vC_1 - 
    #         \pd{{}_{I}\vJ_{P_2}}{t}(t, \vq_2) \vC_2 \right) \\
    #         &=
    #         2 \left(\pd{{}_I \vr_{OP_1}}{t}(t, \vq_1) - 
    #         \pd{{}_I \vr_{OP_2}}{t}(t, \vq_2)\right)^\mathrm{T}
    #         \Big({}_{I}\vJ_{P_1}(t, \vq_1), -{}_{I}\vJ_{P_2}(t, \vq_2)\Big) \\
    #         & + 2 ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2))\T
    #         \Big(\pd{{}_{I}\vJ_{P_1}}{t}(t, \vq_1), -\pd{{}_{I}\vJ_{P_2}}{t}(t, \vq_2)\Big)
    #         \in \mathbb{R}^{1 \times f} \; .

    #     The term $\\frac{\\partial^2 g_i}{\\partial q^j \\partial t}$ is stored in ``g_qt[i, j]``.
    #     """
    #     nq1 = self.body1.n_qDOF
    #     r_OP1 = self.body1.position(t, q[:nq1], self.point_ID_1) 
    #     r_OP2 = self.body2.position(t, q[nq1:], self.point_ID_2)
    #     r_OP1_t = self.body1.position_t(t, q[:nq1], self.point_ID_1) 
    #     r_OP2_t = self.body2.position_t(t, q[nq1:], self.point_ID_2)
    #     J1 = self.body1.position_q(t, q[:nq1], self.point_ID_1) 
    #     J2 = self.body2.position_q(t, q[nq1:], self.point_ID_2)
    #     J1_t = self.body1.position_qt(t, q[:nq1], self.point_ID_1) 
    #     J2_t = self.body2.position_qt(t, q[nq1:], self.point_ID_2)
    #     tmp1 = 2 * (r_OP1_t - r_OP2_t) @ np.hstack([J1,-J2])
    #     tmp2 = 2 * (r_OP1 - r_OP2) @ np.hstack([J1_t,-J2_t])
    #     return tmp1 + tmp2

    # def gap_qtt(self, t, q):
    #     r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates and time

    #     Parameters
    #     ----------
    #     t : float
    #         time instant

    #     q : numpy.ndarray, shape (f,)
    #         generalized coordinates of involved bodies

    #     Returns
    #     -------
    #     g_qtt : numpy.ndarray, shape (1, f)
    #         partial derivatives of gap functions w.r.t. generalized coordinates and time

    #     Notes
    #     -----
    #     Partial derivatives of gap functions w.r.t. generalized coordinates and time

    #     .. math::
    #         \frac{\partial^3 \vg}{\partial \vq \partial t^2} & =
    #         2 \left( \frac{\partial^2 {}_I \vr_{OP_1}}{\partial t^2}(t, \vq_1) - 
    #         \frac{\partial^2 {}_I \vr_{OP_2}}{\partial t^2}(t, \vq_2) \right)^\mathrm{T}
    #         \left({}_{I}\vJ_{P_1}(t, \vq_1) \vC_1 - 
    #         {}_{I}\vJ_{P_2}(t, \vq_2) \vC_2\right)  \\
    #         & + 4 \left(\pd{{}_I \vr_{OP_1}}{t}(t, \vq_1) - 
    #         \pd{{}_I \vr_{OP_2}}{t}(t, \vq_2)\right)^\mathrm{T}
    #         \left(\pd{{}_{I}\vJ_{P_1}}{t}(t, \vq_1) \vC_1 - 
    #         \pd{{}_{I}\vJ_{P_2}}{t}(t, \vq_2) \vC_2 \right) \\
    #         & + 2 ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2))\T
    #         \left(\frac{\partial^2 {}_{I}\vJ_{P_1}}{\partial t^2}(t, \vq_1) \vC_1 - 
    #         \frac{\partial^2 {}_{I}\vJ_{P_2}}{\partial t^2}(t, \vq_2) \vC_2 \right) \\
    #         & = 
    #         2 \left( \frac{\partial^2 {}_I \vr_{OP_1}}{\partial t^2}(t, \vq_1) - 
    #         \frac{\partial^2 {}_I \vr_{OP_2}}{\partial t^2}(t, \vq_2) \right)^\mathrm{T}
    #         \Big({}_{I}\vJ_{P_1}(t, \vq_1), -{}_{I}\vJ_{P_2}(t, \vq_2)\Big) \\
    #         & + 4 \left(\pd{{}_I \vr_{OP_1}}{t}(t, \vq_1) - 
    #         \pd{{}_I \vr_{OP_2}}{t}(t, \vq_2)\right)^\mathrm{T}
    #         \Big(\pd{{}_{I}\vJ_{P_1}}{t}(t, \vq_1), -\pd{{}_{I}\vJ_{P_2}}{t}(t, \vq_2)\Big) \\
    #         & + 2 ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2))\T
    #         \Big(\frac{\partial^2 {}_{I}\vJ_{P_1}}{\partial t^2}(t, \vq_1), - 
    #         \frac{\partial^2 {}_{I}\vJ_{P_2}}{\partial t^2}(t, \vq_2) \Big)
    #         \in \mathbb{R}^{1 \times f} \; .

    #     The term $\\frac{\\partial^3 g_i}{\\partial q^j \\partial t^2}$ is stored in ``g_qtt[i, j]``.
    #     """
    #     nq1 = self.body1.n_qDOF
    #     r_OP1 = self.body1.position(t, q[:nq1], self.point_ID_1) 
    #     r_OP2 = self.body2.position(t, q[nq1:], self.point_ID_2)
    #     r_OP1_t = self.body1.position_t(t, q[:nq1], self.point_ID_1) 
    #     r_OP2_t = self.body2.position_t(t, q[nq1:], self.point_ID_2)
    #     r_OP1_tt = self.body1.position_tt(t, q[:nq1], self.point_ID_1) 
    #     r_OP2_tt = self.body2.position_tt(t, q[nq1:], self.point_ID_2)
    #     J1 = self.body1.position_q(t, q[:nq1], self.point_ID_1) 
    #     J2 = self.body2.position_q(t, q[nq1:], self.point_ID_2)
    #     J1_t = self.body1.position_qt(t, q[:nq1], self.point_ID_1) 
    #     J2_t = self.body2.position_qt(t, q[nq1:], self.point_ID_2)
    #     J1_tt = self.body1.position_qtt(t, q[:nq1], self.point_ID_1) 
    #     J2_tt = self.body2.position_qtt(t, q[nq1:], self.point_ID_2)
    #     tmp1 = 2 * (r_OP1_tt - r_OP2_tt) @ np.hstack([J1,-J2])
    #     tmp2 = 4 * (r_OP1_t - r_OP2_t) @ np.hstack([J1_t,-J2_t])
    #     tmp3 = 2 * (r_OP1 - r_OP2) @ np.hstack([J1_tt,-J2_tt])
    #     return tmp1 + tmp2 + tmp3

    # def gap_qq(self, t, q):
    #     r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates.

    #     Parameters
    #     ----------
    #     t : float
    #         time instant

    #     q : numpy.ndarray, shape (f,)
    #         generalized coordinates of involved bodies

    #     Returns
    #     -------
    #     g_qq : numpy.ndarray, shape (1, f, f)
    #         second partial derivative of gap functions w.r.t. generalized coordinates

    #     Notes
    #     -----
    #     Second partial derivatives of gap functions w.r.t. generalized coordinates

    #     .. math::
    #         \frac{\partial^2 \vg}{\partial \vq^2} = 
    #         2 & \Big({}_I\vJ_{P_1}(t, \vq_1)\T{}_I\vJ_{P_1}(t, \vq_1) + 
    #         ({}_I \vr_{OP_1}(t, \vq_1) - 
    #         {}_I \vr_{OP_2}(t, \vq_2)) \cdot \pd{{}_{I} \vJ_{OP_1}}{\vq_1}(t, \vq_1)\Big) : \vC_1 \otimes \vC_1 \\
    #         - &2 \Big({}_I\vJ_{P_1}(t, \vq_1)\T{}_I\vJ_{P_2}(t, \vq_2)\Big) : \vC_1 \otimes \vC_2 \\
    #         - &2 \Big({}_I\vJ_{P_2}(t, \vq_2)\T{}_I\vJ_{P_1}(t, \vq_1)\Big) : \vC_2 \otimes \vC_1 \\
    #         +&  2 \Big({}_I\vJ_{P_2}(t, \vq_2)\T{}_I\vJ_{P_2}(t, \vq_2) - 
    #         ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2)) \cdot
    #         \pd{{}_{I} \vJ_{OP_2}}{\vq_2}(t, \vq_2)\Big) : \vC_2 \otimes \vC_2 \\
    #         &\in \mathbb{R}^{1 \times f \times f} \; .


    #     The term $\\frac{\\partial^2 g_i}{\\partial q^j \\partial q^k}$ is stored in ``g_qq[i, j, k]``.
    #     """
    #     nq1 = self.body1.n_qDOF
    #     nq2 = self.body2.n_qDOF
    #     nq = nq1 + nq2
    #     r_OP1 = self.body1.position(t, q[:nq1], self.point_ID_1) 
    #     r_OP2 = self.body2.position(t, q[nq1:], self.point_ID_2)
    #     J1 = self.body1.position_q(t, q[:nq1], self.point_ID_1) 
    #     J2 = self.body2.position_q(t, q[nq1:], self.point_ID_2)
    #     J1_q = self.body1.position_qq(t, q[:nq1], self.point_ID_1) 
    #     J2_q = self.body2.position_qq(t, q[nq1:], self.point_ID_2)

    #     g_qq = np.zeros((1, nq, nq))
        
    #     g_qq[0, :nq1, :nq1] =  2 * J1.T @ J1 + 2 * np.tensordot((r_OP1 - r_OP2), J1_q, 1) # g_qq[0, :nq1, :nq1] =  2 * J1.T @ J1 + 2 * (J1_q.T @ (r_OP1 - r_OP2)).T
    #     g_qq[0, :nq1, nq1:] = -2 * J1.T @ J2
    #     g_qq[0, nq1:, :nq1] = -2 * J2.T @ J1
    #     g_qq[0, nq1:, nq1:] =  2 * J2.T @ J2 - 2 * np.tensordot((r_OP1 - r_OP2), J2_q, 1) # g_qq[0, nq1:, nq1:] =  2 * J2.T @ J2 - 2 * (J2_q.T @ (r_OP1 - r_OP2)).T

    #     return g_qq

    # def gap_qqt(self, t, q):
    #     r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates and time.

    #     Parameters
    #     ----------
    #     t : float
    #         time instant

    #     q : numpy.ndarray, shape (f,)
    #         generalized coordinates of involved bodies

    #     Returns
    #     -------
    #     g_qqt : numpy.ndarray, shape (1, f, f)
    #         partial derivatives of gap functions w.r.t. generalized coordinates and time

    #     Notes
    #     -----
    #     Partial derivatives of gap functions w.r.t. generalized coordinates and time

    #     .. math::
    #         \frac{\partial^3 \vg}{\partial \vq^2 \partial t} = 
    #         &2  \Big(\pd{{}_I\vJ_{P_1}}{t}(t, \vq_1)\T{}_I\vJ_{P_1}(t, \vq_1) + 
    #         {}_I\vJ_{P_1}(t, \vq_1)\T\pd{{}_I\vJ_{P_1}}{t}(t, \vq_1) \\ & +
    #         \Big(\pd{{}_I \vr_{OP_1}}{t}(t, \vq_1) - 
    #         \pd{{}_I \vr_{OP_2}}{t}(t, \vq_2)\Big) \cdot \pd{{}_{I} \vJ_{OP_1}}{\vq_1}(t, \vq_1) \\
    #         & + ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2)) \cdot 
    #         \frac{\partial^2 {}_{I} \vJ_{OP_1}}{\partial \vq_1 \partial t}(t, \vq_1)\Big) : \vC_1 \otimes \vC_1 \\
    #         & - 2 \Big(\pd{{}_I\vJ_{P_1}}{t}(t, \vq_1)\T{}_I\vJ_{P_2}(t, \vq_2) + 
    #         {}_I\vJ_{P_1}(t, \vq_1)\T\pd{{}_I\vJ_{P_2}}{t}(t, \vq_2)\Big) : \vC_1 \otimes \vC_2 \\
    #         & - 2 \Big(\pd{{}_I\vJ_{P_2}}{t}(t, \vq_2)\T{}_I\vJ_{P_1}(t, \vq_1) + 
    #         {}_I\vJ_{P_2}(t, \vq_2)\T\pd{{}_I\vJ_{P_1}}{t}(t, \vq_1)\Big)  : \vC_2 \otimes \vC_1 \\
    #         &+ 2 \Big(\pd{{}_I\vJ_{P_2}}{t}(t, \vq_2)\T{}_I\vJ_{P_2}(t, \vq_2) + 
    #         {}_I\vJ_{P_2}(t, \vq_2)\T\pd{{}_I\vJ_{P_2}}{t}(t, \vq_2) \\ & -
    #         \Big(\pd{{}_I \vr_{OP_1}}{t}(t, \vq_1) - 
    #         \pd{{}_I \vr_{OP_2}}{t}(t, \vq_2)\Big) \cdot \pd{{}_{I} \vJ_{OP_2}}{\vq_2}(t, \vq_2) \\
    #         & - ({}_I \vr_{OP_1}(t, \vq_1) - {}_I \vr_{OP_2}(t, \vq_2)) \cdot 
    #         \frac{\partial^2 {}_{I} \vJ_{OP_2}}{\partial \vq_2 \partial t}(t, \vq_2)\Big) : \vC_2 \otimes \vC_2 \\
    #         &\in \mathbb{R}^{1 \times f \times f} \; .

    #     The term $\\frac{\\partial^3 g_i}{\\partial q^j \\partial q^k \\partial t}$ is stored in ``g_qqt[i, j, k]``.
    #     """
    #     nq1 = self.body1.n_qDOF
    #     nq2 = self.body2.n_qDOF
    #     nq = nq1 + nq2
    #     r_OP1 = self.body1.position(t, q[:nq1], self.point_ID_1) 
    #     r_OP2 = self.body2.position(t, q[nq1:], self.point_ID_2)
    #     r_OP1_t = self.body1.position_t(t, q[:nq1], self.point_ID_1) 
    #     r_OP2_t = self.body2.position_t(t, q[nq1:], self.point_ID_2)
    #     J1 = self.body1.position_q(t, q[:nq1], self.point_ID_1) 
    #     J2 = self.body2.position_q(t, q[nq1:], self.point_ID_2)
    #     J1_t = self.body1.position_qt(t, q[:nq1], self.point_ID_1) 
    #     J2_t = self.body2.position_qt(t, q[nq1:], self.point_ID_2)
    #     J1_q = self.body1.position_qq(t, q[:nq1], self.point_ID_1) 
    #     J2_q = self.body2.position_qq(t, q[nq1:], self.point_ID_2)
    #     J1_qt = self.body1.position_qqt(t, q[:nq1], self.point_ID_1) 
    #     J2_qt = self.body2.position_qqt(t, q[nq1:], self.point_ID_2)

    #     g_qqt = np.zeros((1, nq, nq))
        
    #     tmp1 = J1_t.T @ J1 + J1.T @ J1_t
    #     tmp2 = np.tensordot((r_OP1_t - r_OP2_t), J1_q, 1) + np.tensordot((r_OP1 - r_OP2), J1_qt, 1) # tmp2 = (J1_q.T @ (r_OP1_t - r_OP2_t)).T + (J1_qt @ (r_OP1 - r_OP2)).T
    #     tmp3 = J2_t.T @ J2 + J2.T @ J2_t
    #     tmp4 = -np.tensordot((r_OP1_t - r_OP2_t), J2_q, 1) - np.tensordot((r_OP1 - r_OP2), J2_qt, 1) # tmp4 = -(J2_q.T @ (r_OP1_t - r_OP2_t)).T - (J2_qt.T @ (r_OP1 - r_OP2)).T
    #     g_qqt[0, :nq1, :nq1] =  2 * (tmp1 + tmp2)
    #     g_qqt[0, :nq1, nq1:] = -2 * (J1_t.T @ J2 + J1.T @ J2_t)        
    #     g_qqt[0, nq1:, :nq1] = -2 * (J2_t.T @ J1 + J2.T @ J1_t)
    #     g_qqt[0, nq1:, nq1:] =  2 * (tmp3 + tmp4)

    #     return g_qqt

    # def gap_qqq(self, t, q):
    #     r"""Partial derivatives of gap functions. Derivatives w.r.t. generalized coordinates.

    #     Parameters
    #     ----------
    #     t : float
    #         time instant

    #     q : numpy.ndarray, shape (f,)
    #         generalized coordinates of involved bodies

    #     Returns
    #     -------
    #     g_qqq : numpy.ndarray, shape (1, f, f, f)
    #         triple partial derivative of gap functions w.r.t. generalized coordinates

    #     Notes
    #     -----        
    #     Triple partial derivatives of gap functions w.r.t. generalized coordinates. The dimensions of the
    #     individual tuples are $nq1 = \\text{\dim}(\\vq_1)$ and $nq2 = \\text{\dim}(\\vq_2)$.
    #     We use the following abbreviations for the indices: $({}_I\\vr_{OP_1})_i = r^1_i$, 
    #     $({}_I\\vr_{OP_2})_i = r^2_i$, $(\\vq_{1})_i = q^1_i$, $(\\vq_{2})_i = q^2_i$, $({}_I\\vJ_{P_1})_{ij} = J^1_{ij}$,
    #     $({}_I\\vJ_{P_2})_{ij} = J^2_{ij}$.


    #     .. math::
    #         & \frac{\partial^3 g}{\partial q_i \partial q_j \partial q_k} = \\
    #         & \quad\sum_{m = 1}^{nq1} \sum_{n = 1}^{nq1} \sum_{p = 1}^{nq1} \sum_{l = 1}^{2} C^1_{mi}C^1_{nj}C^1_{pk} \, 2 \left[
    #         \pd{J^{1}_{lm}}{q^1_p} J^1_{ln} + J^1_{lm} \pd{J^1_{ln}}{q^1_p}
    #         + J^1_{lp} \pd{J^1_{lm}}{q^1_n} + (r^1_l - r^2_l) \frac{\partial^2 J^1_{lm}}{\partial q^1_n \partial q^1_p}\right]\\
    #         & +\sum_{m = 1}^{nq1} \sum_{n = 1}^{nq1} \sum_{p = 1}^{nq2} \sum_{l = 1}^{2} C^1_{mi}C^1_{nj}C^2_{pk}
    #         \left[-2 J^2_{lp} \pd{J^1_{lm}}{q^1_n} \right] \\
    #         & +\sum_{m = 1}^{nq1} \sum_{n = 1}^{nq2} \sum_{p = 1}^{nq1} \sum_{l = 1}^{2} C^1_{mi}C^2_{nj}C^1_{pk}
    #         \left[-2 \pd{J^1_{lm}}{q^1_p} J^2_{ln} \right] \\
    #         & +\sum_{m = 1}^{nq1} \sum_{n = 1}^{nq2} \sum_{p = 1}^{nq2} \sum_{l = 1}^{2} C^1_{mi}C^2_{nj}C^2_{pk}
    #         \left[-2 J^1_{lm} \pd{J^2_{ln}}{q^2_p}  \right] \\
    #         & +\sum_{m = 1}^{nq2} \sum_{n = 1}^{nq1} \sum_{p = 1}^{nq1} \sum_{l = 1}^{2} C^2_{mi}C^1_{nj}C^1_{pk}
    #         \left[-2 J^2_{lm} \pd{J^1_{ln}}{q^1_p}  \right] \\
    #         & +\sum_{m = 1}^{nq2} \sum_{n = 1}^{nq1} \sum_{p = 1}^{nq2} \sum_{l = 1}^{2} C^2_{mi}C^1_{nj}C^2_{pk}
    #         \left[-2 \pd{J^2_{lm}}{q^2_p}  J^1_{ln}  \right] \\
    #         & +\sum_{m = 1}^{nq2} \sum_{n = 1}^{nq2} \sum_{p = 1}^{nq1} \sum_{l = 1}^{2} C^2_{mi}C^2_{nj}C^1_{pk}
    #         \left[-2 J^1_{lp} \pd{J^2_{lm}}{q^2_n} \right] \\
    #         & + \sum_{m = 1}^{nq2} \sum_{n = 1}^{nq2} \sum_{p = 1}^{nq2} \sum_{l = 1}^{2} C^2_{mi}C^2_{nj}C^2_{pk} \, 2 \left[
    #         \pd{J^{2}_{lm}}{q^2_p} J^2_{ln} + J^2_{lm} \pd{J^2_{ln}}{q^2_p}
    #         + J^2_{lp} \pd{J^2_{lm}}{q^2_n} - (r^1_l - r^2_l) \frac{\partial^2 J^2_{lm}}{\partial q^2_n \partial q^2_p}\right]
            

    #     The term $\\frac{\\partial^3 g_i}{\\partial q^j \\partial q^k \\partial q^l}$ is stored in ``g_qqq[i, j, k, l]``.
    #     """

    #     nq1 = self.body1.n_qDOF
    #     nq2 = self.body2.n_qDOF
    #     nq = nq1 + nq2

    #     dof1 = np.arange(0, nq1)
    #     dof2 = np.arange(nq1, nq)

    #     r_OP1 = self.body1.position(t, q[dof1], self.point_ID_1) 
    #     r_OP2 = self.body2.position(t, q[dof2], self.point_ID_2)
    #     r_OP1_t = self.body1.position_t(t, q[dof1], self.point_ID_1) 
    #     r_OP2_t = self.body2.position_t(t, q[dof2], self.point_ID_2)
    #     J1 = self.body1.position_q(t, q[dof1], self.point_ID_1) 
    #     J2 = self.body2.position_q(t, q[dof2], self.point_ID_2)
    #     J1_t = self.body1.position_qt(t, q[dof1], self.point_ID_1) 
    #     J2_t = self.body2.position_qt(t, q[dof2], self.point_ID_2)
    #     J1_q = self.body1.position_qq(t, q[dof1], self.point_ID_1) 
    #     J2_q = self.body2.position_qq(t, q[dof2], self.point_ID_2)
    #     J1_qt = self.body1.position_qqt(t, q[dof1], self.point_ID_1) 
    #     J2_qt = self.body2.position_qqt(t, q[dof2], self.point_ID_2)
    #     J1_qq = self.body1.position_qqq(t, q[dof1], self.point_ID_1) 
    #     J2_qq = self.body2.position_qqq(t, q[dof2], self.point_ID_2)

    #     g_qqq = np.zeros((1, nq, nq, nq))

    #     for m in range(nq1):
    #         for n in range(nq1):
    #             for p in range(nq1):
    #                 for l in range(2):
    #                     g_qqq[0, dof1[m], dof1[n], dof1[p]] += 2 * (J1_q[l, m, p] * J1[l, n] + J1[l, m] * J1_q[l, n, p] \
    #                                                             + J1[l, p] * J1_q[l, m, n] + (r_OP1[l] - r_OP2[l]) * J1_qq[l, m, n, p])

    #         for n in range(nq1):
    #             for p in range(nq2):
    #                 for l in range(2):                                    
    #                     g_qqq[0, dof1[m], dof1[n], dof2[p]] += - 2 * J2[l, p] * J1_q[l, m, n]

    #         for n in range(nq2):
    #             for p in range(nq1):
    #                 for l in range(2):                          
    #                     g_qqq[0, dof1[m], dof2[n], dof1[p]] += -2 * J1_q[l, m, p] * J2[l, n]

    #         for n in range(nq2):
    #             for p in range(nq2):
    #                 for l in range(2): 
    #                     g_qqq[0, dof1[m], dof2[n], dof2[p]] += -2 * J1[l, m] * J2_q[l, n, p]

    #     for m in range(nq2):
    #         for n in range(nq1):
    #             for p in range(nq1):
    #                 for l in range(2): 
    #                     g_qqq[0, dof2[m], dof1[n], dof1[p]] += -2 * J2[l, m] * J1_q[l, n, p]

    #         for n in range(nq1):
    #             for p in range(nq2):
    #                 for l in range(2):
    #                     g_qqq[0, dof2[m], dof1[n], dof2[p]] += -2 * J2_q[l, m, p] * J1[l, n]

    #         for n in range(nq2):
    #             for p in range(nq1):
    #                 for l in range(2): 
    #                     g_qqq[0, dof2[m], dof2[n], dof1[p]] += -2 * J1[l, p] * J2_q[l, m, n]

    #         for n in range(nq2):
    #             for p in range(nq2):
    #                 for l in range(2): 
    #                     g_qqq[0, dof2[m], dof2[n], dof2[p]] += 2 * (J2_q[l, m, p] * J2[l, n] + J2[l, m] * J2_q[l, n, p] \
    #                                                             + J2[l, p] * J2_q[l, m, n] - (r_OP1[l] - r_OP2[l]) * J2_qq[l, m, n, p])      

    #     return g_qqq