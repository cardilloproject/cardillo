from time import time
import vtk
from vtk import vtkDataSetMapper, vtkActor, vtkGeometryFilter
from cardillo.solver import Solution
from .vtk_export import make_ugrid
from ..rods._base_export import RodExportBase


class Renderer:
    def __init__(self, system, contributions=None, winsize=(1000, 1000)) -> None:
        self.active = False
        self.system = system
        self.contributions = (
            system.contributions if contributions is None else contributions
        )
        self.__renderer = vtk.vtkRenderer()
        self.__renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("DarkGreen"))

        self.fps_actor = vtk.vtkTextActor()
        self.__renderer.AddActor(self.fps_actor)
        self.__init_fps()

        self.renwin = vtk.vtkRenderWindow()
        self.renwin.SetWindowName("")
        self.renwin.AddRenderer(self.__renderer)
        self.renwin.MakeRenderWindowInteractor()
        self.renwin.SetSize(*winsize)
        self.interactor = self.renwin.GetInteractor()
        self.interactor.AddObserver(
            vtk.vtkCommand.ExitEvent, self.__handle_window_closed
        )
        # self.observer = self.interactor.AddObserver(
        #     vtk.vtkCommand.windowevent, self.__handle_window_closed
        # )
        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.__renderer)
        self.cam_widget.On()
        self.camera = self.__renderer.GetActiveCamera()

        for contr in self.contributions:
            if hasattr(contr, "actors"):
                for actor in contr.actors:
                    self.__renderer.AddActor(actor)

    def __update_fps(self):
        self.n_frame += 1
        self.fps = self.n_frame / (time() - self.t0)
        self.fps_actor.SetInput(
            f" frame {self.n_frame} / {self.tot_frame}, fps {self.fps:.2f}" + " " * 10
        )

    def __init_fps(self, tot_frame=0):
        self.t0 = time()
        self.tot_frame = tot_frame
        self.n_frame = 0
        self.fps_actor.SetInput("")

    def __handle_window_closed(self, inter, event):
        print("stop app")
        self.stop_step_render()
        self.interactor.TerminateApp()

    def render_solution(self, solution, repeat=False):
        self.active = True
        while True:
            self.__init_fps(len(solution.t))
            for sol_i in solution:
                self.__update_fps()
                self.render_sol_i(sol_i)
                if not self.active:
                    return
            if not repeat:
                self.active = False
                return

    def render_sol_i(self, sol_i):
        for contr in self.contributions:
            if hasattr(contr, "export"):
                points, cells, point_data, cell_data = contr.export(sol_i)
                ugrid = make_ugrid(points, cells, point_data, cell_data)
                if not hasattr(contr, "_vtkfilter"):
                    contr._vtkfilter = vtkGeometryFilter()
                    if isinstance(contr, RodExportBase):
                        contr._vtkfilter.SetNonlinearSubdivisionLevel(3)
                    mapper = vtkDataSetMapper()
                    actor = vtkActor()
                    actor.SetMapper(mapper)
                    self.__renderer.AddActor(actor)
                    mapper.SetInputConnection(contr._vtkfilter.GetOutputPort())
                contr._vtkfilter.SetInputData(ugrid)
        self.renwin.Render()
        self.interactor.ProcessEvents()

    def stop_step_render(self):
        if self.active:
            self.active = False

    def start_step_render(self):
        self.active = True

        def decorate_step_callback(system_step_callback):
            def __step_callback(t, q, u):
                if self.active:
                    self.tot_frame += 1
                    self.__update_fps()
                    self.render_sol_i(Solution(self.system, t, q, u))
                return system_step_callback(t, q, u)

            return __step_callback

        self.system.step_callback = decorate_step_callback(self.system.step_callback)
        self.__init_fps()
