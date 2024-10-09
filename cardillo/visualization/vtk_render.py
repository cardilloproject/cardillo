from sys import platform
from time import time
from multiprocessing import Process, Queue, Event
import vtk
from vtk import vtkDataSetMapper, vtkActor
from cardillo.solver import Solution
from .vtk_export import make_ugrid


class RendererBase:
    def __init__(self, system, contributions=None, winsize=(1000, 1000)) -> None:
        self.active = False
        self.system = system
        self.contributions = (
            system.contributions if contributions is None else contributions
        )
        self.ren = vtk.vtkRenderer()
        # self.ren.SetBackground(vtkNamedColors().GetColor3d("Grey"))
        self.ren.SetBackground(vtk.vtkNamedColors().GetColor3d("DarkGreen"))

        self.fps_actor = vtk.vtkTextActor()
        self.ren.AddActor(self.fps_actor)
        self._init_fps()

        self.renwin = vtk.vtkRenderWindow()
        self.renwin.SetWindowName("")
        self.renwin.AddRenderer(self.ren)
        self.renwin.MakeRenderWindowInteractor()
        self.renwin.SetSize(*winsize)
        self.interactor = self.renwin.GetInteractor()
        self.observer = self.interactor.AddObserver(
            vtk.vtkCommand.ExitEvent, self.__handle_window_closed
        )
        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.ren)
        self.cam_widget.On()
        self.camera = self.ren.GetActiveCamera()

        for contr in self.contributions:
            if hasattr(contr, "actors"):
                for actor in contr.actors:
                    self.ren.AddActor(actor)

    def _update_fps(self):
        self.n_frame += 1
        self.fps = self.n_frame / (time() - self.t0)
        self.fps_actor.SetInput(
            f" frame {self.n_frame} / {self.tot_frame}, fps {self.fps:.2f}" + " " * 10
        )

    def _init_fps(self, tot_frame=0):
        self.t0 = time()
        self.tot_frame = tot_frame
        self.n_frame = 0
        self.fps_actor.SetInput("")

    def __handle_window_closed(self, inter, event):
        self.stop_step_render()
        self.interactor.TerminateApp()

    def render_solution(self, solution, repeat=False):
        self.active = True
        while True:
            self._init_fps(len(solution.t))
            for sol_i in solution:
                self._update_fps()
                self.step_render(sol_i.t, sol_i.q, sol_i.u)
                if not self.active:
                    return
            if not repeat:
                self.active = False
                return

    def step_render(self, t, q, u):
        for contr in self.contributions:
            if hasattr(contr, "step_render"):
                contr.step_render(t, q[contr.qDOF], u[contr.uDOF])
            # TODO: rod needs nonlinear subdivision filter
            elif hasattr(contr, "export"):
                points, cells, point_data, cell_data = contr.export(
                    Solution(self.system, t, q, u)
                )
                ugrid = make_ugrid(points, cells, point_data, cell_data)
                if not hasattr(contr, "vtk_data_map"):
                    contr.vtk_data_map = vtkDataSetMapper()
                    actor = vtkActor()
                    actor.SetMapper(contr.vtk_data_map)
                    self.ren.AddActor(actor)
                contr.vtk_data_map.SetInputData(ugrid)
        self.renwin.Render()
        self.interactor.ProcessEvents()

    def start_interaction(self, t, q, u):
        self._init_fps()
        self.step_render(t, q, u)
        self.interactor.Start()


class RendererSync(RendererBase):
    def __init__(self, system, contributions=None, winsize=(1000, 1000)) -> None:
        super().__init__(system, contributions, winsize)

    def start_step_render(self):
        self.active = True

        def decorate_step_callback(system_step_callback):
            def __step_callback(t, q, u):
                if self.active:
                    self.tot_frame += 1
                    self._update_fps()
                    self.step_render(t, q, u)
                return system_step_callback(t, q, u)

            return __step_callback

        self.system.step_callback = decorate_step_callback(self.system.step_callback)
        self._init_fps()

    def stop_step_render(self):
        if self.active:
            self.active = False


class RendererLinux(RendererBase):
    def __init__(self, system, contributions=None, winsize=(1000, 1000)) -> None:
        super().__init__(system, contributions, winsize)
        self.queue = Queue()
        self.exit_event = Event()

        def target(queue, iterrupt):
            self._init_fps()
            while True:
                el = queue.get()
                if el is None:
                    return
                elif iterrupt.is_set():
                    while self.queue.qsize():
                        self.queue.get()
                    return
                self.tot_frame += 1
                self._update_fps()
                self.step_render(*el)

        self.process = Process(target=target, args=(self.queue, self.exit_event))

    def start_step_render(self):
        self.active = True

        def decorate_step_callback(system_step_callback):
            def __step_callback(t, q, u):
                if self.active:
                    self.queue.put((t, q, u))
                return system_step_callback(t, q, u)

            return __step_callback

        self.system.step_callback = decorate_step_callback(self.system.step_callback)

        self.exit_event.clear()
        self.process.start()

    def stop_step_render(self, wait=False):
        if self.active:
            self.active = False
            if self.process.is_alive():
                self.queue.put(None)
                if wait:
                    self.process.join()
                else:
                    self.exit_event.set()
