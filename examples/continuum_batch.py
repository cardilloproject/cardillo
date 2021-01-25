from threading import main_thread
import numpy as np
import os
import dill
import datetime
import pathlib

from cardillo.discretization.mesh2D import Mesh2D, rectangle
from cardillo.discretization.B_spline import Knot_vector
from cardillo.discretization.indexing import split2D
from cardillo.model.continuum import Ogden1997_compressible, First_gradient, Ogden1997_complete_2D_incompressible
from cardillo.model.continuum import Pantographic_sheet, verify_derivatives, Pantographic_lattice, Bipantographic_fabric
from cardillo.solver import Newton
from cardillo.model import Model
from cardillo.math.algebra import A_IK_basic_z, norm2

def init_guess(continuum, t, t_new, x):
    # computes initial guess for new displacement step
    # shifts all nodes within surface in dependence of displacement of left and right edge
    left_DOF = continuum.mesh.edge_qDOF[0]
    right_DOF = continuum.mesh.edge_qDOF[1]
    z = continuum.z(t, x)
    z_new = continuum.z(t_new, x)
    left_z = z[left_DOF.T]
    right_z = z[right_DOF.T]

    # compute left and right displacement
    left_disp = z_new[left_DOF.T] - left_z 
    right_disp = z_new[right_DOF.T] - right_z 

    # iterate over control points
    for a in range(continuum.mesh.nn):
        a_xi, a_eta = split2D(a, (continuum.mesh.nn_xi,))
        
        # x and y value of control points
        z_a = z[np.array([a, a + continuum.mesh.nn])]

        # compute distance to left and right points
        left_a = left_z[a_eta]
        right_a = right_z[a_eta]
        d_left =  norm2(z_a - left_a)
        d_right =  norm2(z_a - right_a)

        p_right = d_left / (d_left + d_right)
        p_left = 1 - p_right
        z_new[np.array([a, a + continuum.mesh.nn])] = z_new[np.array([a, a + continuum.mesh.nn])] + left_disp[a_eta] * p_left + right_disp[a_eta] * p_right
    return z_new[continuum.fDOF]


def first_gradient_solve(case="test_a",  n_load_steps=20, source="Treolar",  element_shape=(45,15)):
    time_string = datetime.datetime.now().strftime("%y%m%d_%H_%M_%S")
    filename = f"{time_string}__{source}_{case}_nstep{n_load_steps}__elshape{element_shape[0]}_{element_shape[1]}"
    folderpath = pathlib.Path("output") / filename  #os.path.join("output", filename)
    folderpath.mkdir(parents=True) #os.makedirs(folderpath)
    filepath = folderpath / (filename + ".dill") #os.path.join(folderpath, filename + ".dill")

    # build mesh
    degrees = (2, 2)
    QP_shape = (3, 3)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)
    
    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=2, basis='B-spline', nq_n=2)

    if source == "Treolar":
        # reference configuration is a rectangle
        B = 10 * np.sqrt(2) * 0.0048
        L = 3 * B

        rectangle_shape = (L, B)
        Z = rectangle(rectangle_shape, mesh, Greville=True)
        z0 = rectangle(rectangle_shape, mesh, Greville=True, Fuzz=0.0)

        # material model
        al = [1.3, 5, -2]    # values: see Ogden 1997 p. 498
        #mu = [6.3, 0.012, -0.1]
        mu_star = [1.491, 0.003, -0.0237]
        mu = [mu_star[i] *4.225 for i in range(len(al))]
        mat = Ogden1997_complete_2D_incompressible(mu, al)

    rectangle_shape = (L, B)
    Z = rectangle(rectangle_shape, mesh, Greville=True)
    z0 = rectangle(rectangle_shape, mesh, Greville=True, Fuzz=0.00001)

    # boundary conditions
    cDOF, b = standard_displacements(mesh, Z, case=case, source=source)

    # 3D continuum
    continuum = First_gradient(None, mat, mesh, Z, z0=z0, cDOF=cDOF, b=b)
    model = Model()
    model.add(continuum)
    model.assemble()

    load_steps = np.linspace(0, 1, n_load_steps)
    tol = 1.0e-6
    max_iter = 12
    init_guess_continuum = lambda t, t_new, x: init_guess(continuum, t, t_new, x)
    solver = Newton(model, n_load_steps=n_load_steps, load_steps=load_steps, tol=tol, max_iter=max_iter, init_guess=init_guess_continuum)
    sol = solver.solve()

    with filepath.open(mode='wb') as f:
        dill.dump((continuum, sol), f)

    first_gradient_post(filepath)


def pantographic_sheet_solve(case="test_a", n_load_steps=20, starting_step=0, source="Maurin", element_shape=(15,5)):
    time_string = datetime.datetime.now().strftime("%y%m%d_%H_%M_%S")
    filename = f"{time_string}__{source}_{case}_nstep{n_load_steps}__elshape{element_shape[0]}_{element_shape[1]}"
    folderpath = pathlib.Path("output") / filename  #os.path.join("output", filename)
    folderpath.mkdir(parents=True) #os.makedirs(folderpath)
    filepath = folderpath / (filename + ".dill") #os.path.join(folderpath, filename + ".dill")

    # build mesh
    degrees = (2, 2)
    QP_shape = (3, 3)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)
    
    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=2, basis='B-spline', nq_n=2)

    if source == "Maurin":
        # reference configuration is a rectangle
        B = 10 * np.sqrt(2) * 0.0048
        L = 3 * B

        rectangle_shape = (L, B)
        Z = rectangle(rectangle_shape, mesh, Greville=True)
        z0 = rectangle(rectangle_shape, mesh, Greville=True, Fuzz=0.00001)

        # material model
        K_rho =  1.34e5 
        K_Gamma = 1.59e2
        K_Theta_s = 1.92e-2
        gamma = 1.36
        # mat = Maurin2019(K_rho, K_Gamma, K_Theta_s, gamma)
        mat_param = (K_rho, K_Gamma, K_Theta_s, gamma)


    elif source == "MaurinSquare":
        # reference configuration is a rectangle
        B = 10 * np.sqrt(2) * 0.0048
        L = 3 * B

        rectangle_shape = (L, B)
        Z = rectangle(rectangle_shape, mesh, Greville=True)
        z0 = rectangle(rectangle_shape, mesh, Greville=True, Fuzz=0.00001)

        # material model
        K_rho =  1.34e5 
        K_Gamma = 1.59e2
        K_Theta_s = 1.92e-2
        gamma = 2 # this is different to previous case
        mat_param = (K_rho, K_Gamma, K_Theta_s, gamma)

    elif source == "Barchiesi" or source == "Barchiesi_newFormat":
            # reference configuration is a rectangle
        B = 7 * 0.017
        L = 11 * 0.017

        if source == "Barchiesi_newFormat":
            B = 10 * np.sqrt(2) * 0.0048
            L = 3 * B

        # material model
        gamma = np.pi / 6
        K_F = 0.9   #[J]
        K_E = 0.33  #[J]
        K_S = 34    #[J]
        mat_param = (gamma, K_F, K_E, K_S)

    rectangle_shape = (L, B)
    Z = rectangle(rectangle_shape, mesh, Greville=True)
    z0 = rectangle(rectangle_shape, mesh, Greville=True, Fuzz=0.00001)

    # boundary conditions
    cDOF, b = standard_displacements(mesh, Z, case=case, source=source)

    # 2D continuum
    if source == "Maurin" or source == "MaurinSquare":
        continuum = Pantographic_lattice(None, mat_param, mesh, Z, z0=z0, cDOF=cDOF, b=b)
    elif source == "Barchiesi" or source == "Barchiesi_newFormat":
        continuum = Bipantographic_fabric(None, mat_param, mesh, Z, z0=z0, cDOF=cDOF, b=b)

    # run derivative check for the chosen material
    verify_derivatives(continuum)

    model = Model()
    model.add(continuum)
    model.assemble()

    init_guess_continuum = lambda t, t_new, x: init_guess(continuum, t, t_new, x)

    load_steps = np.linspace(0, 1, n_load_steps)[starting_step:]
    # test init_guess
    continuum.q0 = init_guess_continuum(0, load_steps[0], z0[continuum.fDOF]) # assembler callback
    
    tol = 1.0e-6
    max_iter = 12
    solver = Newton(model, n_load_steps=n_load_steps, load_steps=load_steps, tol=tol, max_iter=max_iter, init_guess=init_guess_continuum)

    sol = solver.solve()
    with filepath.open(mode='wb') as f:
        dill.dump((continuum, sol), f)

    pantographic_sheet_post(filepath)

def first_gradient_post(filepath):
    filepath = pathlib.Path(filepath)
    with filepath.open(mode='rb') as f:
        continuum, sol = dill.load(f)
    continuum.post_processing(sol.t, sol.q, filepath.parent / filepath.stem)

def pantographic_sheet_post(filepath, project_to_reference=False, case=None, last_only=False):
    # carry out vtu export
    filepath = pathlib.Path(filepath)
    with filepath.open(mode='rb') as f:
        continuum, sol = dill.load(f)

    if not last_only:
        continuum.post_processing(sol.t, sol.q, filepath.parent / filepath.stem, project_to_reference=project_to_reference)
    else:
        continuum.post_processing(sol.t[::len(sol.t)-1], sol.q[::len(sol.t)-1], filepath.parent / filepath.stem, project_to_reference=project_to_reference)

def standard_displacements(mesh, Z, case, source="Maurin"):
    from cardillo.math.algebra import A_IK_basic_z

    # recreates the test cases A, B, C, D as defined in dellIsola 2016. The dimensions of the undeformed rectangle must correspond to the source as well.
    if case == "test_a":
        displacement = (0.0567, 0)
        if source == "Barchiesi":
            displacement = (0.05, 0)

        cDOF1 = mesh.edge_qDOF[0].ravel()
        cDOF2x = mesh.edge_qDOF[1][0]
        cDOF2y = mesh.edge_qDOF[1][1]
        cDOF2 = np.concatenate((cDOF2x, cDOF2y))
        cDOF = np.concatenate((cDOF1, cDOF2)) 

        b1 = lambda t: Z[cDOF1]
        b2x = lambda t: Z[cDOF2x] + t * displacement[0]
        b2y = lambda t: Z[cDOF2y]
        b = lambda t: np.concatenate((b1(t), b2x(t), b2y(t)))

    elif case == "test_b":
        displacement = (0, 0.9* 20 * np.sqrt(2) * 0.0048)

        cDOF1 = mesh.edge_qDOF[0].ravel()
        cDOF2x = mesh.edge_qDOF[1][0]
        cDOF2y = mesh.edge_qDOF[1][1]
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

        cDOF1 = mesh.edge_qDOF[0].ravel()
        cDOF2x = mesh.edge_qDOF[1][0]
        cDOF2y = mesh.edge_qDOF[1][1]
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

        cDOF1x = mesh.edge_qDOF[0][0]
        cDOF1y = mesh.edge_qDOF[0][1]        
        cDOF2x = mesh.edge_qDOF[1][0]
        cDOF2y = mesh.edge_qDOF[1][1]
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

    if case == "simple_tension":#
        displacement = (5* 3 * 10 * np.sqrt(2) * 0.0048, 0)
       
        cDOF1x = mesh.edge_qDOF[0][0] # x direcion, one node
        cDOF2x = mesh.edge_qDOF[1][0]
        cDOF1y = mesh.edge_qDOF[0][1][0] # y direcion, one node
        # cDOF2 = np.hstack((cDOF2x, cDOF2y))
        cDOF = np.hstack((cDOF1x, cDOF2x, cDOF1y)) 

        b1x = lambda t: Z[cDOF1x]
        b2x = lambda t: Z[cDOF2x] + t * displacement[0]
        b1y = lambda t: Z[cDOF1y]
        b = lambda t: np.hstack((b1x(t), b2x(t), b1y(t)))

    if case == "equibiaxial_tension":#
        displacement = (3* 3 * 10 * np.sqrt(2) * 0.0048, 3 * 10 * np.sqrt(2) * 0.0048)
       
        cDOF1x = mesh.edge_qDOF[0][0] # x direcion, one node
        cDOF2x = mesh.edge_qDOF[1][0]
        cDOF3y = mesh.edge_qDOF[2][1]
        cDOF4y = mesh.edge_qDOF[3][1]
        # cDOF2 = np.hstack((cDOF2x, cDOF2y))
        cDOF = np.hstack((cDOF1x, cDOF2x, cDOF3y, cDOF4y)) 

        b1x = lambda t: Z[cDOF1x]
        b2x = lambda t: Z[cDOF2x] + t * displacement[0]
        b3y = lambda t: Z[cDOF3y]
        b4y = lambda t: Z[cDOF4y] + t * displacement[1]
        b = lambda t: np.hstack((b1x(t), b2x(t), b3y(t),  b4y(t)))

    if case == "simple_shear":#
        displacement = (0, 3 * 3 * 10 * np.sqrt(2) * 0.0048)
       
        cDOF1x = mesh.edge_qDOF[0][0][1:-1] # x direcion, one node
        cDOF2x = mesh.edge_qDOF[1][0][1:-1]
        cDOF3x = mesh.edge_qDOF[2][0]
        cDOF4x = mesh.edge_qDOF[3][0]
        cDOF1y = mesh.edge_qDOF[0][1][1:-1]
        cDOF2y = mesh.edge_qDOF[1][1][1:-1]
        cDOF3y = mesh.edge_qDOF[2][1]
        cDOF4y = mesh.edge_qDOF[3][1]
        # cDOF2 = np.hstack((cDOF2x, cDOF2y))
        cDOF = np.hstack((cDOF1x, cDOF2x, cDOF3x, cDOF4x, cDOF1y,  cDOF2y,  cDOF3y, cDOF4y)) 

        linear_increase = (Z[cDOF3x] - Z[cDOF3x][0])/(Z[cDOF3x][-1] - Z[cDOF3x][0])

        b1x = lambda t: Z[cDOF1x]
        b2x = lambda t: Z[cDOF2x] 
        b3x = lambda t: Z[cDOF3x] 
        b4x = lambda t: Z[cDOF4x] 
        b1y = lambda t: Z[cDOF1y] 
        b2y = lambda t: Z[cDOF2y] + t * displacement[1]
        b3y = lambda t: Z[cDOF3y] + t * displacement[1] * linear_increase
        b4y = lambda t: Z[cDOF4y] + t * displacement[1] * linear_increase

        b = lambda t: np.hstack((b1x(t), b2x(t), b3x(t), b4x(t), b1y(t), b2y(t), b3y(t), b4y(t)))

    if case == "pure_shear":
        displacement = (3* 3 * 10 * np.sqrt(2) * 0.0048, 0)
       
        cDOF1x = mesh.edge_qDOF[0][0] # x direcion, one node
        cDOF2x = mesh.edge_qDOF[1][0]
        cDOF3y = mesh.edge_qDOF[2][1]
        cDOF4y = mesh.edge_qDOF[3][1]
        # cDOF2 = np.hstack((cDOF2x, cDOF2y))
        cDOF = np.hstack((cDOF1x, cDOF2x, cDOF3y, cDOF4y)) 

        b1x = lambda t: Z[cDOF1x]
        b2x = lambda t: Z[cDOF2x] + t * displacement[0]
        b3y = lambda t: Z[cDOF3y]
        b4y = lambda t: Z[cDOF4y] 
        b = lambda t: np.hstack((b1x(t), b2x(t), b3y(t),  b4y(t)))

    return cDOF, b

def task_queue_solve():
    pantographic_sheet_solve(case="test_a", n_load_steps=5, starting_step=2, source="Maurin", element_shape=(6, 3))
    # pantographic_sheet_solve(case="test_a", n_load_steps=60, starting_step=2, source="Maurin", element_shape=(60, 60))
    # pantographic_sheet_solve(case="test_a", n_load_steps=30, starting_step=0, source="Barchiesi_newFormat", element_shape=(30, 10))
    # pantographic_sheet_solve(case="test_d", n_load_steps=90, starting_step=0, source="Barchiesi_newFormat", element_shape=(45, 15))
    # pantographic_sheet_solve(case="test_c", n_load_steps=90, starting_step=0, source="Barchiesi_newFormat", element_shape=(45, 15))

    # first_gradient_solve(case="simple_tension", n_load_steps=20, source="Treolar", element_shape=(15, 5))
    # first_gradient_solve(case="pure_shear", n_load_steps=20, source="Treolar", element_shape=(12, 4))
    # first_gradient_solve(case="equibiaxial_tension", n_load_steps=20, source="Treolar", element_shape=(12, 4))
    # first_gradient_solve(case="test_a", n_load_steps=20, source="Treolar", element_shape=(30, 10))

def task_queue_post():
    # pantographic_sheet_post(r"output\1109_11_56_55__MaurinSquare_test_b_nstep30__elshape30_10_40percent\1109_11_56_55__MaurinSquare_test_b_nstep30__elshape30_10.dill", case="Maurin")
   pass

if __name__ == "__main__":
    task_queue_solve()
    # task_queue_post()