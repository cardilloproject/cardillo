import numpy as np

from cardillo import TEST_FOLDER, System
from cardillo.contacts import Sphere2Plane
from cardillo.discrete import PointMass
from cardillo.forces import Force
from cardillo.solver import BackwardEuler
from cardillo.visualization import Export

g = 10


def create_subsystems(nx, ny, dist, mass, frame, height):
    pms, grav, contacts = [], [], []
    for ix in range(nx):
        for iy in range(ny):
            pms.append(
                PointMass(
                    mass=mass,
                    q0=np.array((ix * dist, iy * dist, height)),
                    name=f"point_mass_{ix}_{iy}",
                )
            )
            grav.append(
                Force(np.array((0, 0, -mass * g)), pms[-1], name=f"force_{ix}_{iy}")
            )
            contacts.append(
                Sphere2Plane(
                    frame, pms[-1], mu=0, r=0, e_N=0.2, name=f"contact_{ix}_{iy}"
                )
            )
    return pms, grav, contacts


def test_vtk_export(delete_files=True):
    nx, ny = 3, 3
    dist = 0.5
    mass = 1
    height = 1

    system = System()
    frame = system.origin

    PMs, grav_forces, contacts = create_subsystems(nx, ny, dist, mass, frame, height)

    system.add(*PMs, *grav_forces, *contacts)
    system.assemble()

    t1 = 3
    dt = 1e-2
    solver = BackwardEuler(system, t1, dt)
    solution = solver.solve()

    ############
    # VTK export
    ############
    path = f"{TEST_FOLDER}/_data"
    folder_name = "test_vtk_export"
    e = Export(path, folder_name, overwrite=True, fps=60, solution=solution)
    e.export_contr(contacts)
    e.export_contr(PMs, name="PMs")
    e.export_contr(frame)
    e.export_contr(grav_forces)

    if delete_files:
        from shutil import rmtree

        rmtree(f"{path}/{folder_name}")


if __name__ == "__main__":
    test_vtk_export(delete_files=False)
