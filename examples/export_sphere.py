from cardillo.visualization import vtk_sphere
import meshio

if __name__ == "__main__":
    radius = 2
    points, cell, point_data = vtk_sphere(radius)

    mesh = meshio.Mesh(
        points,
        [cell],
        point_data=point_data,
    )
    mesh.write("sphere.vtu", binary=False)
