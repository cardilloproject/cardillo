from math import pi
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection, Spherical
from cardillo.forces import B_Moment, Force
from cardillo.math import e2, e3
from cardillo.rods import (
    CircularCrossSection,
    RectangularCrossSection,
    Simo1986,
    animate_beam,
)
from cardillo.rods.cosseratRod import (
    make_CosseratRod,
)
from cardillo.rods._director import make_CosseratRodDirectorBubnovGalerkin
from cardillo.solver import Newton, SolverOptions
from cardillo.utility.sensor import Sensor, SensorRecords


""" Derived cantilever beam example. 
Purpose is to show the different exports. And to be an example how to define the export of rods.
"""


def cantilever(
    Rod,  # TODO: add type hint
    constitutive_law=Simo1986,  # TODO: add type hint
    *,
    nelements: int = 10,
    #
    n_load_steps: int = 10,
    load_type: str = "moment",
    #
    VTK_export: bool = False,
    name: str = "simulation",
):
    # handle name
    plot_name = name.replace("_", " ")
    save_name = name.replace(" ", "_")

    ############
    # parameters
    ############
    # geometry of the rod
    length = 2 * np.pi

    # cross section properties only for visualization purposes
    slenderness = 1.0e1
    width = length / slenderness
    cross_section_rect = RectangularCrossSection(width, width)
    cross_section_circle = CircularCrossSection(width / 2, export_as_wedge=False)
    cross_section_circle_wedge = CircularCrossSection(width / 2, export_as_wedge=True)

    # material properties
    Ei = np.array([5, 1, 1])
    Fi = np.array([0.5, 2, 2])

    material_model = constitutive_law(Ei, Fi)

    # initialize system
    system = System()

    #####
    # rod
    #####
    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(nelements, length)
    # construct cantilever
    cantilever = Rod(
        cross_section_rect,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
    )

    ##########
    # clamping
    ##########
    # clamping = RigidConnection(system.origin, cantilever, xi2=0)
    # system.add(cantilever, clamping)
    # markers = [
    #     Sensor(cantilever, xi=0.0, name="sensor_start"),
    #     Sensor(cantilever, xi=0.5, name="sensor_middle"),
    #     Sensor(cantilever, xi=1.0, name="sensor_end"),
    # ]
    # system.add(*markers)
    clamping_left = Spherical(system.origin, cantilever, np.zeros(3), xi2=0)
    r_OP = np.array([length, 0.0, 0.0])
    frame = Frame(r_OP=r_OP)
    clamping_right = Spherical(frame, cantilever, r_OP, xi2=1)
    system.add(cantilever, frame, clamping_left, clamping_right)

    ###############
    # applied loads
    ###############
    if load_type == "moment":
        # moment at cantilever tip
        m = material_model.Fi[2] * 2 * np.pi / length
        M = lambda t: t * e3 * m
        moment = B_Moment(M, cantilever, 1)
        system.add(moment)
    elif load_type == "constant_end_load":
        # spatially fixed load at cantilever tip
        P = lambda t: material_model.Fi[2] * (10 * t) / length**2
        F = lambda t: -P(t) * e2 * 0.1
        force = Force(F, cantilever, 3 / 4)
        system.add(force)
    else:
        raise NotImplementedError

    # assemble system
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    ############
    # simulation
    ############
    options = SolverOptions(
        numerical_jacobian_method="2-point",
    )
    solver = Newton(system, n_load_steps=n_load_steps, options=options)  # create solver
    sol = solver.solve()  # solve static equilibrium equations

    # animation
    t = sol.t
    q = sol.q
    animate_beam(t, q, [cantilever], scale=length)

    #################
    # post-processing
    #################

    # # VTK export
    # dir_name = Path(__file__).parent
    # records = [SensorRecords.r_OP, SensorRecords.A_IB]
    # [m.save(dir_name, "csv", sol, records) for m in markers]
    # if VTK_export:
    #     from copy import deepcopy

    #     # export with rectangular cross-section
    #     rod_volume_rectangular = deepcopy(cantilever)
    #     rod_volume_rectangular.name = "cantilever_volume_rectangular"
    #     rod_volume_rectangular._export_dict["level"] = "volume"
    #     rod_volume_rectangular._export_dict["stresses"] = True
    #     rod_volume_rectangular._export_dict["volume_directors"] = True
    #     system.add(rod_volume_rectangular)

    #     # export with circular cross-section (hexagonal cells)
    #     rod_volume_circle = deepcopy(cantilever)
    #     rod_volume_circle.name = "cantilever_volume_circle"
    #     rod_volume_circle.cross_section = cross_section_circle
    #     rod_volume_circle._export_dict["level"] = "volume"
    #     rod_volume_circle._export_dict["stresses"] = True
    #     rod_volume_circle._export_dict["volume_directors"] = True
    #     rod_volume_circle._export_dict["surface_normals"] = True
    #     system.add(rod_volume_circle)

    #     # export with circular cross-section (wedge cells)
    #     rod_volume_circle_wedge = deepcopy(cantilever)
    #     rod_volume_circle_wedge.name = "cantilever_volume_circle_wedge"
    #     rod_volume_circle_wedge.cross_section = cross_section_circle_wedge
    #     rod_volume_circle_wedge._export_dict["level"] = "volume"
    #     rod_volume_circle_wedge._export_dict["stresses"] = True
    #     rod_volume_circle_wedge._export_dict["volume_directors"] = True
    #     rod_volume_circle_wedge._export_dict["surface_normals"] = True
    #     system.add(rod_volume_circle_wedge)

    #     # export only nodal quantities for fast export (rectangle)
    #     rod_NodalVolume = deepcopy(cantilever)
    #     rod_NodalVolume.name = "cantilever_NodalVolume"
    #     rod_NodalVolume._export_dict["level"] = "NodalVolume"
    #     system.add(rod_NodalVolume)

    #     # export only nodal quantities for fast export (circle)
    #     rod_NodalVolume_circle = deepcopy(cantilever)
    #     rod_NodalVolume_circle.cross_section = cross_section_circle
    #     rod_NodalVolume_circle.name = "cantilever_NodalVolume_circle"
    #     rod_NodalVolume_circle._export_dict["level"] = "NodalVolume"
    #     system.add(rod_NodalVolume_circle)

    #     # export only nodal quantities for fast export (circle as wedge)
    #     rod_NodalVolume_circle_wedge = deepcopy(cantilever)
    #     rod_NodalVolume_circle_wedge.cross_section = cross_section_circle_wedge
    #     rod_NodalVolume_circle_wedge.name = "cantilever_NodalVolume_circle_wedge"
    #     rod_NodalVolume_circle_wedge._export_dict["level"] = "NodalVolume"
    #     system.add(rod_NodalVolume_circle_wedge)

    #     # export only centerline & directors
    #     cantilever.name = "cantilever"
    #     cantilever._export_dict["level"] = "centerline + directors"
    #     # this rod is already in the system

    #     # add rods and export
    #     system.export(dir_name, f"vtk/{save_name}", sol)


if __name__ == "__main__":
    cantilever(
        # Rod=make_CosseratRod(interpolation="SE3", mixed=True, constraints=[0, 1, 2]),
        # Rod=make_CosseratRod(interpolation="R12", mixed=True, constraints=[0, 1, 2]),
        # Rod=make_CosseratRod(
        #     mixed=True, polynomial_degree=1
        # ),  # , constraints=[0, 1, 2]),
        Rod=make_CosseratRodDirectorBubnovGalerkin(
            polynomial_degree=1,
        ),
        # load_type="moment",
        load_type="constant_end_load",
        nelements=20,
        VTK_export=True,
        name="Cosserat mixed",
    )
