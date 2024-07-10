from vtk import (
    vtkRenderer,
    vtkRenderWindow,
    vtkCommand,
    vtkCameraOrientationWidget,
    vtkNamedColors,
    vtkActor,
    vtkDataSetMapper,
)
from cardillo.solver import Solution
from .vtk_export import make_ugrid


class VTKRenderer:
    def __init__(self, system) -> None:
        self.active = False
        self.system = system
        self.ren = vtkRenderer()
        self.ren.SetBackground(vtkNamedColors().GetColor3d("Grey"))
        self.renwin = vtkRenderWindow()
        self.renwin.AddRenderer(self.ren)
        self.renwin.MakeRenderWindowInteractor()
        self.renwin.SetSize(800, 800)
        self.interactor = self.renwin.GetInteractor()
        self.observer = self.interactor.AddObserver(
            vtkCommand.ExitEvent, self.__handle_window_closed
        )
        self.cam = vtkCameraOrientationWidget()
        self.cam.SetParentRenderer(self.ren)
        self.cam.On()

        def decorate_step_callback(fun):
            def __step_callback(t, q, u):
                self.step_render(t, q, u)
                return fun(t, q, u)

            return __step_callback

        self.system.step_callback = decorate_step_callback(self.system.step_callback)

    def step_render(self, t_i, q_i, u_i):
        sol_i = Solution(self.system, t_i, q_i, u_i)
        if not self.active:
            return
        for contr in self.system.contributions:
            if hasattr(contr, "update_vtk_tf"):
                if hasattr(contr, "vtk_tf"):
                    contr.update_vtk_tf(sol_i)
                else:
                    self.ren.AddActor(contr.update_vtk_tf(sol_i))
            elif hasattr(contr, "export"):
                points, cells, point_data, cell_data = contr.export(sol_i)
                ugrid = make_ugrid(points, cells, point_data, cell_data)
                if not hasattr(contr, "vtk_data_map"):
                    contr.vtk_data_map = vtkDataSetMapper()
                    actor = vtkActor()
                    actor.SetMapper(contr.vtk_data_map)
                    self.ren.AddActor(actor)
                contr.vtk_data_map.SetInputData(ugrid)
            else:
                continue
        self.renwin.Render()
        self.interactor.ProcessEvents()

    def __handle_window_closed(self, inter, event):
        self.active = False
        self.interactor.TerminateApp()

    def start_step_render(self):
        self.active = True

    def start_interaction(self):
        self.interactor.RemoveObserver(self.observer)
        self.interactor.Start()

    def render_solution(self, solution, repeat=False):
        self.active = True
        while self.active:
            for sol_i in solution:
                self.step_render(sol_i.t, sol_i.q, sol_i.u)
                if not self.active:
                    break
            if not repeat:
                break
