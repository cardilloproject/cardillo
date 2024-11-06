from pathlib import Path
from shutil import rmtree
from xml.dom import minidom
import numpy as np
import vtk
from cardillo.solver import Solution


# https://github.com/nschloe/meshio/wiki/Node-ordering-in-cells
# https://examples.vtk.org/site/Cxx/GeometricObjects/IsoparametricCellsDemo/
# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
# TODO: check if all possible cases are implemented
def dtype_map(dtype: np.dtype):
    # possible vtk arrays: https://vtk.org/doc/nightly/html/classvtkAOSDataArrayTemplate.html
    if np.issubdtype(dtype, np.floating):
        return vtk.vtkDoubleArray
    elif np.issubdtype(dtype, np.integer):
        return vtk.vtkIntArray
    elif np.issubdtype(dtype, np.bool_):
        return vtk.vtkBitArray


def make_ugrid(points, cells, point_data, cell_data):
    ugrid = vtk.vtkUnstructuredGrid()

    # points
    vtkpoints = vtk.vtkPoints()
    vtkpoints.Allocate(len(points))
    for p in points:
        vtkpoints.InsertNextPoint(p)
    ugrid.SetPoints(vtkpoints)

    # cells
    ugrid.Allocate(len(cells))
    for cell_type, connectivity in cells:
        ugrid.InsertNextCell(cell_type, len(connectivity), connectivity)

    # point data
    pdata = ugrid.GetPointData()
    if point_data is not None:
        for key, value in point_data.items():
            value = np.array(value)
            n, dim = value.shape
            parray = dtype_map(value.dtype)()
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
            m, dim = value.shape
            carray = dtype_map(value.dtype)()
            carray.SetName(key)
            carray.SetNumberOfTuples(m)
            carray.SetNumberOfComponents(dim)
            for i, vi in enumerate(value):
                carray.InsertTuple(i, vi)

            if key == "HigherOrderDegrees":
                cdata.SetHigherOrderDegrees(carray)
            else:
                cdata.AddArray(carray)
    return ugrid


class Export:
    def __init__(
        self,
        path: Path,
        folder_name: str,
        overwrite: bool,
        fps: float,
        solution: Solution,
        write_ascii=False,
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

    def __prepare_data(self, solution):
        frames = len(solution.t)
        # target_frames = min(len(t), 100)
        animation_time_ = solution.t[-1] - solution.t[0]
        target_frames = max(1, int(animation_time_ * self.fps))
        frac = max(1, int(frames / target_frames))

        frames = target_frames
        keys = [*solution.__dict__.keys()]
        keys.remove("solver_summary")
        keys.remove("system")

        new_solution = {}
        for key in keys:
            try:
                new_solution[key] = solution.__getattribute__(key)[::frac]
            except:
                new_solution[key] = None
        self.solution = Solution(
            system=solution.system,
            solver_summary=solution.solver_summary,
            **new_solution,
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
                    data_write[key] = np.vstack((data_write[key], data_read[key]))

    def __export_list(self, sol_i, **kwargs):
        contr_list = kwargs.pop("contr_list")
        points, cells, point_data, cell_data = [], [], {}, {}
        l = 0
        for contr in contr_list:
            p, c, p_data, c_data = contr.export(sol_i, **kwargs)
            l = len(points)
            points.extend(p)

            cells.extend([(tup[0], [idx + l for idx in tup[1]]) for tup in c])
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

            ugrid = make_ugrid(points, cells, point_data, cell_data)

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
