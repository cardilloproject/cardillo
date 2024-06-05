import numpy as np
from xml.dom import minidom
from pathlib import Path
from shutil import rmtree
import vtk
from cardillo.solver import Solution

# https://github.com/nschloe/meshio/wiki/Node-ordering-in-cells
# https://examples.vtk.org/site/Cxx/GeometricObjects/IsoparametricCellsDemo/
# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
cell_map = {
    "vertex": vtk.VTK_VERTEX,
    "line": vtk.VTK_LINE,
    "triangle": vtk.VTK_TRIANGLE,
    "quad": vtk.VTK_QUAD,
    "quad8": vtk.VTK_QUADRATIC_QUAD,
    "tetra": vtk.VTK_TETRA,
    "hexahedron": vtk.VTK_HEXAHEDRON,
    "hexahedron20": vtk.VTK_QUADRATIC_HEXAHEDRON,
    "hexahedron24": vtk.VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON,
    "wedge": vtk.VTK_WEDGE,
    "pyramid": vtk.VTK_PYRAMID,
    "line3": vtk.VTK_QUADRATIC_EDGE,
    "triangle6": vtk.VTK_QUADRATIC_TRIANGLE,
    "quad9": vtk.VTK_BIQUADRATIC_QUAD,
    "tetra10": vtk.VTK_QUADRATIC_TETRA,
    "hexahedron27": vtk.VTK_TRIQUADRATIC_HEXAHEDRON,
    "pyramid13": vtk.VTK_QUADRATIC_PYRAMID,
    "pyramid14": vtk.VTK_TRIQUADRATIC_PYRAMID,
    "line4": vtk.VTK_CUBIC_LINE,
    "VTK_BEZIER_CURVE": vtk.VTK_BEZIER_CURVE,
    "VTK_BEZIER_TRIANGLE": vtk.VTK_BEZIER_TRIANGLE,
    "VTK_BEZIER_QUADRILATERAL": vtk.VTK_BEZIER_QUADRILATERAL,
    "VTK_BEZIER_TETRAHEDRON": vtk.VTK_BEZIER_TETRAHEDRON,
    "VTK_BEZIER_HEXAHEDRON": vtk.VTK_BEZIER_HEXAHEDRON,
    "VTK_BEZIER_WEDGE": vtk.VTK_BEZIER_WEDGE,
    "VTK_BEZIER_PYRAMID": vtk.VTK_BEZIER_PYRAMID,
}

# dtype_map = {
#     np.float128: vtk.vtkDoubleArray,
# }

# # print(dtype_map[np.int32])
# print(dtype_map[np.float128])

# exit()

# cell_name = "VTK_BEZIER_WEDGE"
# var = eval("vtk." + cell_name)


class Export:
    def __init__(
        self,
        path: Path,
        folder_name: str,
        overwrite: bool,
        fps: float,
        solution: Solution,
        write_ascii=True,
    ) -> None:
        self.path = path
        self.folder = self.__create_vtk_folder(folder_name, overwrite)
        self.fps = fps
        self.system = solution.system
        self.write_ascii = (write_ascii,)
        self.__prepare_data(solution)

    # helper functions
    def __vtk_file(self):
        self.root = minidom.Document()

        self.vtk_file = self.root.createElement("VTKFile")
        self.vtk_file.setAttribute("type", "Collection")
        self.root.appendChild(self.vtk_file)

        self.collection = self.root.createElement("Collection")
        self.vtk_file.appendChild(self.collection)

    def __unique_file_name(self, file_name):
        file_name_ = file_name
        i = 1
        while (self.path / f"{file_name_}.pvd").exists():
            file_name_ = f"{file_name}{i}"
            i += 1
        return file_name_

    def __write_time_step_and_name(self, t, file):
        # write time step and file name in pvd file
        dataset = self.root.createElement("DataSet")
        dataset.setAttribute("timestep", f"{t:0.6f}")
        dataset.setAttribute("file", file.name)
        self.collection.appendChild(dataset)

    def _write_pvd_file(self, path):
        xml_str = self.root.toprettyxml(indent="\t")
        with (path).open("w") as f:
            f.write(xml_str)

    def __prepare_data(self, sol):
        frames = len(sol.t)
        # target_frames = min(len(t), 100)
        animation_time_ = sol.t[-1] - sol.t[0]
        target_frames = max(1, int(animation_time_ * self.fps))
        frac = max(1, int(frames / target_frames))

        frames = target_frames
        t = sol.t[::frac]
        q = sol.q[::frac]
        if sol.u is not None:
            u = sol.u[::frac]
        else:
            u = None
        if sol.u_dot is not None:
            u_dot = sol.u_dot[::frac]
        else:
            u_dot = None
        if sol.la_g is not None:
            la_g = sol.la_g[::frac]
        else:
            la_g = None
        if sol.la_gamma is not None:
            la_gamma = sol.la_gamma[::frac]
        else:
            la_gamma = None
        if hasattr(sol, "q_dot"):
            q_dot = sol.q_dot[::frac]
        else:
            q_dot = None
        if hasattr(sol, "P_N"):
            P_N = sol.P_N[::frac]
        else:
            P_N = None
        if hasattr(sol, "P_F"):
            P_F = sol.P_F[::frac]
        else:
            P_F = None

        # TODO default values + not None values of solution object
        self.solution = Solution(
            sol.system,
            t=t,
            q=q,
            u=u,
            q_dot=q_dot,
            u_dot=u_dot,
            la_g=la_g,
            la_gamma=la_gamma,
            P_N=P_N,
            P_F=P_F,
        )

    def __create_vtk_folder(self, folder_name: str, overwrite: bool):
        path = Path(self.path, folder_name)
        i = 0
        if not overwrite:
            while path.exists():
                path = self.path / str(folder_name + f"_{i}")
                i += 1
        else:
            if path.exists():
                rmtree(path)
        path.mkdir(parents=True, exist_ok=overwrite)
        self.path = path
        return path.stem

    def __add_key(self, data_read, data_write):
        for key in data_read.keys():
            if not key in data_write.keys():
                data_write[key] = data_read[key]
            else:
                if isinstance(data_read[key], list):
                    data_write[key].extend(data_read[key])
                else:
                    data_write[key] = np.hstack((data_write[key], data_read[key]))

    def __export_list(self, sol_i, **kwargs):
        contr_list = kwargs.pop("contr_list")
        points, cells, point_data, cell_data = [], [], {}, {}
        l = 0
        for contr in contr_list:
            p, c, p_data, c_data = contr.export(sol_i, **kwargs)
            l = len(points)
            points.extend(p)
            cells.extend(
                [(tup[0], [[i + l for i in idx] for idx in tup[1]]) for tup in c]
            )
            if c_data is not None:
                self.__add_key(c_data, cell_data)
            if p_data is not None:
                self.__add_key(p_data, point_data)

        return points, cells, point_data, cell_data

    def __set_contr_name(self, contr, kwargs):
        if "file_name" in kwargs and kwargs["file_name"] is not None:
            contr_name = kwargs["file_name"]
        else:
            contr_name = contr.name
        return contr_name

    def export_contr(self, contr, **kwargs):
        """_summary_

        Args:
            contr (Any): one contribution or list, tuple, dict... of same type
            base_export (bool): kwargs arg, decides if derived class uses export function of base class or overriden function in derived class
            file_name (string): kwargs arg, set custom file name instead of class name of exported contr
        """
        self.__vtk_file()

        # export one contr
        if not isinstance(contr, (list, tuple, np.ndarray)):
            contr_name = self.__set_contr_name(contr, kwargs)
            export = contr.export
        # export list of contributions of same type (mixing types is not useful)
        else:
            contr_name = self.__set_contr_name(contr[0], kwargs)
            export = self.__export_list
            kwargs["contr_list"] = contr
        file_name = self.__unique_file_name(contr_name)
        for i, sol_i in enumerate(self.solution):
            file_i = self.path / f"{file_name}_{i}.vtu"
            self.__write_time_step_and_name(sol_i.t, file_i)

            points, cells, point_data, cell_data = export(sol_i, **kwargs)

            ugrid = vtk.vtkUnstructuredGrid()

            # points
            vtkpoints = vtk.vtkPoints()
            vtkpoints.Allocate(len(points))
            for p in points:
                vtkpoints.InsertNextPoint(p)
            ugrid.SetPoints(vtkpoints)

            # cells = [
            #     # ("VTK_LINE", [
            #     #     [0, 1],
            #     #     [1, 2],
            #     # ]),
            #     # TODO: Implement this approach
            #     ("VTK_LINE", [0, 1]),
            #     ("VTK_LINE", [1, 2]),
            #     ("VTK_QUAD", [[], []]),
            # ]

            # cells
            # TODO: This allocation is wrong!
            ugrid.Allocate(len(cells))
            for cell_type, connectivity in cells:
                vtktype = cell_map[cell_type]
                # this will be replace since we directly use something like
                # from vtk import VTK_LINE
                # vtktype = eval("vtk." + cell_type)
                for con in connectivity:
                    ugrid.InsertNextCell(vtktype, len(con), con)

            # point data
            pdata = ugrid.GetPointData()
            if point_data is not None:
                for key, value in point_data.items():
                    value = np.atleast_2d(value)
                    n, dim = value.shape

                    # TODO: Mapping numpy.dtype => vtkDataArray
                    # parray = dtype_map[value.dtype]
                    parray = vtk.vtkDoubleArray()
                    parray.SetName(key)
                    parray.SetNumberOfTuples(n)
                    parray.SetNumberOfComponents(dim)
                    for i, vi in enumerate(value):
                        parray.InsertTuple(i, vi)

                    if key == "RationalWeights":
                        pdata.SetRationalWeights(parray)
                    else:
                        pdata.AddArray(parray)

            # cell data
            cdata = ugrid.GetCellData()
            if cell_data is not None:
                for key, value in cell_data.items():
                    value = np.array(value)
                    # TODO: I think the first dimension can be removed here but
                    # has to be implemented in all export functions.
                    n, m, dim = value.shape

                    for i, vi in enumerate(value):
                        # TODO: Mapping numpy.dtype => vtkDataArray
                        # carray = dtype_map[value.dtype]
                        carray = vtk.vtkDoubleArray()
                        carray.SetName(key)
                        carray.SetNumberOfTuples(m)
                        carray.SetNumberOfComponents(dim)
                        for j, vij in enumerate(vi):
                            carray.InsertTuple(j, vij)

                        if key == "HigherOrderDegrees":
                            cdata.SetHigherOrderDegrees(carray)
                        else:
                            cdata.AddArray(carray)

            # write data
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetInputData(ugrid)
            writer.SetFileName(file_i)
            if self.write_ascii:
                writer.SetDataModeToAscii()
            else:
                writer.SetDataModeToBinary()
            writer.Write()

        self._write_pvd_file(self.path / f"{file_name}.pvd")
