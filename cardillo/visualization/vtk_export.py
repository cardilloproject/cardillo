import numpy as np
import meshio
from xml.dom import minidom
from pathlib import Path
from shutil import rmtree

from cardillo.solver import Solution


class Export:
    def __init__(
        self,
        path: Path,
        folder_name: str,
        overwrite: bool,
        fps: float,
        solution: Solution,
    ) -> None:
        self.path = path
        self.folder = self.__create_vtk_folder(folder_name, overwrite)
        self.fps = fps
        self.system = solution.system
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
        file_name_ = f"{file_name}0"
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
            if hasattr(contr, "pre_iteration_update"):
                contr.pre_iteration_update(
                    sol_i.t, sol_i.q[contr.qDOF], sol_i.u[contr.uDOF]
                )
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
            contr_name = contr.__class__.__name__
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

            meshio.write_points_cells(
                filename=file_i,
                points=points,
                cells=cells,
                point_data=point_data,
                cell_data=cell_data,
                # binary=False,
            )
        self._write_pvd_file(self.path / f"{file_name}.pvd")
