from threading import main_thread
import numpy as np
import matplotlib.pyplot as plt

from cardillo.discretization.mesh import Mesh, cube
from cardillo.discretization.mesh2D import Mesh2D, rectangle
from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume
from cardillo.discretization.indexing import flat3D
from cardillo.model.continuum import Ogden1997_compressible, First_gradient
from cardillo.model.continuum import Pantographic_sheet, Maurin2019_linear, Maurin2019, verify_derivatives
from cardillo.solver import Newton, Generalized_alpha_1, Euler_backward
from cardillo.model import Model
from cardillo.math.algebra import A_IK_basic_z
from cardillo.model.force_distr2D import Force_distr2D
from cardillo.model.force_distr3D import Force_distr3D

def test_cube():
    TractionForce = False
    Gravity = False
    Statics = False
    # build mesh
    # degrees = (2, 2, 2)
    # QP_shape = (3, 3, 3)
    # # element_shape = (5, 5, 5)
    # element_shape = (2, 2, 2)
    degrees = (1, 1, 1)
    QP_shape = (2, 2, 2)
    element_shape = (3, 3, 3)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    mesh = Mesh(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=3)

    # reference configuration is a cube
    L = 1
    B = 1
    H = 2
    cube_shape = (L, B, H)
    Z = cube(cube_shape, mesh, Greville=True)

    # material model
    mu1 = 0.3 # * 1e3
    mu2 = 0.5 # * 1e3
    mat = Ogden1997_compressible(mu1, mu2)

    density = 1e-2

    if Statics:
        # boundary conditions
        if TractionForce:
            # cDOF = mesh.surface_qDOF[0].reshape(-1)
            cDOF = mesh.surface_qDOF[2].reshape(-1)
            # cDOF = mesh.surface_qDOF[4].reshape(-1)
            b = lambda t: Z[cDOF]

        else:
            # cDOF1 = mesh.surface_qDOF[4].reshape(-1)
            # cDOF2 = mesh.surface_qDOF[5][2]
            # cDOF = np.concatenate((cDOF1, cDOF2))
            # b1 = lambda t: Z[cDOF1]
            # b2 = lambda t: Z[cDOF2] + t * 0.5
            # b = lambda t: np.concatenate((b1(t), b2(t)))
            cDOF = mesh.surface_qDOF[4].ravel()
            b = lambda t: Z[cDOF]
    else:
        cDOF_xi = mesh.surface_qDOF[4][0]
        cDOF_eta = mesh.surface_qDOF[4][1]
        cDOF_zeta = mesh.surface_qDOF[4][2]
        cDOF = np.concatenate((cDOF_xi, cDOF_eta, cDOF_zeta))
        Omega = 2 * np.pi
        b_xi = lambda t: Z[cDOF_xi] + 0.1 * np.sin(Omega * t)
        b_eta = lambda t: Z[cDOF_eta]
        b_zeta = lambda t: Z[cDOF_zeta]
        b = lambda t: np.concatenate((b_xi(t), b_eta(t), b_zeta(t)))


    # 3D continuum
    continuum = First_gradient(density, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)
    # continuum = First_gradient(density, mat, mesh, Z)


    # build model
    model = Model()
    model.add(continuum)

    if TractionForce:
        # F = lambda t, xi, eta: t * np.array([-2.5e0, 0, 0]) * (0.25 - (xi-0.5)**2) * (0.25 - (eta-0.5)**2)
        # model.add(Force_distr2D(F, continuum, 1))
        # F = lambda t, xi, eta: t * np.array([0, -2.5e0, 0]) * (0.25 - (xi-0.5)**2) * (0.25 - (eta-0.5)**2)
        # model.add(Force_distr2D(F, continuum, 5))
        F = lambda t, xi, eta: np.array([0, 0, -5e0]) * (0.25 - (xi-0.5)**2) * (0.25 - (eta-0.5)**2)
        model.add(Force_distr2D(F, continuum, 5))
    
    if Gravity:
        if Statics:
            G = lambda t, xi, eta, zeta: t * np.array([0, 0, -9.81 * density])
        else:
            G = lambda t, xi, eta, zeta: np.array([0, 0, -9.81 * density])
        model.add(Force_distr3D(G, continuum))

    model.assemble()

    # M = model.M(0, model.q0)
    # np.set_printoptions(precision=5, suppress=True)
    # print(M.toarray())
    # print(np.linalg.det(M.toarray()))

    if Statics:
    # static solver
        n_load_steps = 10
        tol = 1.0e-5
        max_iter = 10
        solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)

    else:
        t1 = 10
        dt = 1e-1
        # solver = Generalized_alpha_1(model, t1, dt=dt, variable_dt=False, rho_inf=0.25)
        solver = Euler_backward(model, t1, dt)

    # import cProfile, pstats
    # pr = cProfile.Profile()
    # pr.enable()
    sol = solver.solve()
    # pr.disable()

    # sortby = 'cumulative'
    # ps = pstats.Stats(pr).sort_stats(sortby)
    # ps.print_stats(0.1) # print only first 10% of the list

    # plt.plot(sol.t, sol.q[:, -1])
    # plt.show()
    # exit()

    # vtk export
    continuum.post_processing(sol.t, sol.q, 'cube')

def test_cylinder():    
    # build mesh
    degrees = (3, 3, 3)
    QP_shape = (3, 3, 3)
    element_shape = (2, 2, 4)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    mesh = Mesh(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=3)
    
    def cylinder(xi, eta, zeta, R=1, H=3):
        xi_ = 2 * xi - 1
        eta_ = 2 * eta - 1

        if np.abs(xi_) > np.abs(eta_):
            r = np.sqrt(1 + eta_**2)
        else:
            r = np.sqrt(1 + xi_**2)

        x = R / r * xi_
        y = R / r * eta_
        z = zeta * H
        return x, y, z

    nxi, neta, nzeta = 10, 10, 10
    xi = np.linspace(0, 1, num=nxi)
    eta = np.linspace(0, 1, num=neta)
    zeta = np.linspace(0, 1, num=nzeta)
    
    n3 = nxi * neta * nzeta
    knots = np.zeros((n3, 3))
    Pw = np.zeros((n3, 3))
    for i, xii in enumerate(xi):
        for j, etai in enumerate(eta):
            for k, zetai in enumerate(zeta):
                idx = flat3D(i, j, k, (nxi, neta, nzeta))
                knots[idx] = xii, etai, zetai
                Pw[idx] = cylinder(xii, etai, zetai)

    cDOF_ = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    X, Y, Z_ = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF_)
    Z = np.concatenate((X, Y, Z_))

    # material model    
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2)

    # boundary conditions
    cDOF1 = mesh.surface_DOF[0].reshape(-1)
    cDOF2 = mesh.surface_DOF[1].reshape(-1)
    cDOF = np.concatenate((cDOF1, cDOF2))
    b1 = lambda t: Z[cDOF1]

    def b2(t, phi0=np.pi, h=0.25):
        cDOF2_xyz = cDOF2.reshape(3, -1).T
        out = np.zeros_like(Z)

        phi = t * phi0
        R = A_IK_basic_z(phi)

        th = t * np.array([0, 0, h])
        for DOF in cDOF2_xyz:
            out[DOF] = R @ Z[DOF] + th
        
        return out[cDOF2]

    b = lambda t: np.concatenate((b1(t), b2(t)))

    # 3D continuum
    continuum = First_gradient(mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    # build model
    model = Model()
    model.add(continuum)
    model.assemble()

    # static solver
    n_load_steps = 10
    tol = 1.0e-5
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)
    
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    sol = solver.solve()
    pr.disable()

    sortby = 'cumulative'
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(0.1) # print only first 10% of the list

    # vtk export
    continuum.post_processing(sol.t, sol.q, 'cylinder')

def test_rectangle():
    # build mesh
    degrees = (1, 1)
    QP_shape = (3, 3)
    element_shape = (4, 8)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)
    
    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=2)

    # reference configuration is a cube
    L = 2
    B = 4

    rectangle_shape = (L, B)
    Z = rectangle(rectangle_shape, mesh, Greville=True)

    # material model    
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2, dim=2)

    # boundary conditions
    cDOF1 = mesh.edge_DOF[0].reshape(-1)
    cDOF2 = mesh.edge_DOF[1][1]
    cDOF = np.concatenate((cDOF1, cDOF2))
    b1 = lambda t: Z[cDOF1]
    b2 = lambda t: Z[cDOF2] + t * 4
    b = lambda t: np.concatenate((b1(t), b2(t)))

    # 3D continuum
    continuum = First_gradient(1, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    

    # vtk export reference configuration
    # continuum.post_processing_single_configuration(0, Z, 'rectangleReferenceConfig.vtu')

    # build model
    model = Model()
    model.add(continuum)
    model.assemble()

    # static solver
    n_load_steps = 30
    tol = 1.0e-5
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)
    
    # import cProfile, pstats
    # pr = cProfile.Profile()
    # pr.enable()
    sol = solver.solve()
    # pr.disable()

    # sortby = 'cumulative'
    # ps = pstats.Stats(pr).sort_stats(sortby)
    # ps.print_stats(0.1) # print only first 10% of the list

    # # vtk export
    continuum.post_processing(sol.t, sol.q, 'rectangle')

def write_xml():
    # write paraview PVD file, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
    from xml.dom import minidom
    
    root = minidom.Document()
    
    vkt_file = root.createElement('VTKFile')
    vkt_file.setAttribute('type', 'Collection')
    root.appendChild(vkt_file)
    
    collection = root.createElement('Collection')
    vkt_file.appendChild(collection)

    for i in range(10):
        ti = 0.1 * i
        dataset = root.createElement('DataSet')
        dataset.setAttribute('timestep', f'{ti:0.6f}')
        # dataset.setAttribute('group', '')
        # dataset.setAttribute('part', '0')
        dataset.setAttribute('file', f'continuum{i}.vtu')
        collection.appendChild(dataset)

    
    xml_str = root.toprettyxml(indent ="\t")  
    
    save_path_file = "continuum.pvd"
    
    with open(save_path_file, "w") as f:
        f.write(xml_str)

def pantographic_sheet():
    # build mesh
    degrees = (2, 2)
    QP_shape = (3, 3)
    element_shape = (4, 3)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)
    
    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=2, basis='B-spline', nq_n=2)

    # reference configuration is a rectangle
    # L = 2
    # B = 2
    B = 10 * np.sqrt(2) * 0.0048
    L = 3 * B

    rectangle_shape = (L, B)
    Z = rectangle(rectangle_shape, mesh, Greville=True)

    # material model
    K_rho = 1.34e5
    K_Gamma = 1.59e2
    K_Theta_s = 1.92e-2
    gamma = 1.36
    mat = Maurin2019_linear(K_rho, K_Gamma, K_Theta_s)
    # mat = Maurin2019(K_rho, K_Gamma, K_Theta_s, gamma)

    
    verify_derivatives(mat)

    # boundary conditions
    # cDOF1 = mesh.edge_DOF[0].ravel()
    # cDOF2x = mesh.edge_DOF[1][0]
    # cDOF2y = mesh.edge_DOF[1][1]
    # cDOF2 = np.concatenate((cDOF2x, cDOF2y))
    # cDOF = np.concatenate((cDOF1, cDOF2))
    # b1 = lambda t: Z[cDOF1]
    # b2x = lambda t: Z[cDOF2x] + t * 0.
    # b2y = lambda t: Z[cDOF2y]
    # b = lambda t: np.concatenate((b1(t), b2x(t), b2y(t)))

    cDOF, b = standard_displacements(mesh, Z, case="test_a")

    # 3D continuum
    continuum = Pantographic_sheet(None, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    model = Model()
    model.add(continuum)
    model.assemble()

    # f_pot = model.f_pot(0, model.q0)
    
    # static solver
    n_load_steps = 2
    tol = 1.0e-5
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)

    sol = solver.solve()

    # vtk export
    continuum.post_processing(sol.t, sol.q, 'pantographic_sheet')


def standard_displacements(mesh, Z, case, displacement=None):
    from cardillo.math.algebra import A_IK_basic_z

    # recreates the test cases A, B, C, D as defined in dellIsola 2016. The dimensions of the undeformed rectangle must correspond to the source as well.
    if case == "test_a":
        displacement = (0.0567, 0)

        cDOF1 = mesh.edge_DOF[0].ravel()
        cDOF2x = mesh.edge_DOF[1][0]
        cDOF2y = mesh.edge_DOF[1][1]
        cDOF2 = np.concatenate((cDOF2x, cDOF2y))
        cDOF = np.concatenate((cDOF1, cDOF2)) 

        b1 = lambda t: Z[cDOF1]
        b2x = lambda t: Z[cDOF2x] + t * displacement[0]
        b2y = lambda t: Z[cDOF2y]
        b = lambda t: np.concatenate((b1(t), b2x(t), b2y(t)))

    elif case == "test_b":
        displacement = (0, 20 * np.sqrt(2) * 0.0048)

        cDOF1 = mesh.edge_DOF[0].ravel()
        cDOF2x = mesh.edge_DOF[1][0]
        cDOF2y = mesh.edge_DOF[1][1]
        cDOF2 = np.concatenate((cDOF2x, cDOF2y))
        cDOF = np.concatenate((cDOF1, cDOF2)) 

        b1 = lambda t: Z[cDOF1]
        b2x = lambda t: Z[cDOF2x] 
        b2y = lambda t: Z[cDOF2y] + t * displacement[1]
        b = lambda t: np.concatenate((b1(t), b2x(t), b2y(t)))

    elif case == "test_c":
        # rotation of right edge in counterclockwise direction around center, translation in x-direction
        alpha = np.pi /4
        translation =  np.array([5 * np.sqrt(2) * 0.0048, 0])
        R = A_IK_basic_z(alpha)[:2, :2]

        cDOF1 = mesh.edge_DOF[0].ravel()
        cDOF2x = mesh.edge_DOF[1][0]
        cDOF2y = mesh.edge_DOF[1][1]
        cDOF2 = np.concatenate((cDOF2x, cDOF2y))
        cDOF = np.concatenate((cDOF1, cDOF2)) 

        # center: pivot point of right edge
        center = np.array([np.mean([Z[cDOF2x[0]], Z[cDOF2x[-1]]]), np.mean([Z[cDOF2y[0]], Z[cDOF2y[-1]]])])
        displacement = R @ (Z[np.array([cDOF2x, cDOF2y])] - center[:,None]) + center[:, None] - Z[np.array([cDOF2x, cDOF2y])] + translation[:,None] 

        b1 = lambda t: Z[cDOF1]
        b2x = lambda t: Z[cDOF2x] + t * displacement[0]
        b2y = lambda t: Z[cDOF2y] + t * displacement[1]
        b = lambda t: np.concatenate((b1(t), b2x(t), b2y(t)))

    elif case == "test_d":
        # rotation of left edge in clockwise direction around top left corner and right edge in counterclockwise direction around top right corner, with overlayed translation
        alpha = np.pi /3
        translation =  np.array([-15 * np.sqrt(2) * 0.0048, 0])
        R1 = A_IK_basic_z(-alpha)[:2, :2]
        R2 = A_IK_basic_z(alpha)[:2, :2]        

        cDOF1x = mesh.edge_DOF[0][0]
        cDOF1y = mesh.edge_DOF[0][1]        
        cDOF2x = mesh.edge_DOF[1][0]
        cDOF2y = mesh.edge_DOF[1][1]
        cDOF1 = np.concatenate((cDOF1x, cDOF1y))
        cDOF2 = np.concatenate((cDOF2x, cDOF2y))
        cDOF = np.concatenate((cDOF1, cDOF2)) 

        # center: pivot point of the edges
        center1 = np.array([Z[cDOF1x[-1]], Z[cDOF1y[-1]]])
        center2 = np.array([Z[cDOF2x[-1]], Z[cDOF2y[-1]]])
        displacement1 = R1 @ (Z[np.array([cDOF1x, cDOF1y])] - center1[:,None]) + center1[:, None] - Z[np.array([cDOF1x, cDOF1y])]
        displacement2 = R2 @ (Z[np.array([cDOF2x, cDOF2y])] - center2[:,None]) + center2[:, None] - Z[np.array([cDOF2x, cDOF2y])] + translation[:,None]

        # b1 = lambda t: Z[cDOF1]
        b1x = lambda t: Z[cDOF1x] + t * displacement1[0]
        b1y = lambda t: Z[cDOF1y] + t * displacement1[1]
        b2x = lambda t: Z[cDOF2x] + t * displacement2[0]
        b2y = lambda t: Z[cDOF2y] + t * displacement2[1]
        b = lambda t: np.concatenate((b1x(t), b1y(t), b2x(t), b2y(t)))

    return cDOF, b

if __name__ == "__main__":
    # test_cube()
    # test_cylinder()
    # test_rectangle()
    # write_xml()
    pantographic_sheet()