# TODO: Discuss if all solvers use relative tolerances.
class SolverSummary:
    def __init__(self, solver_name):
        self.solver_name = solver_name
        self.fixed_point_n_iter_list = []
        self.fixed_point_abs_errors = []
        self.newton_n_iter_list = []
        self.newton_abs_errors = []
        self.n_lu = 0

    def clear(self):
        self.fixed_point_n_iter_list = []
        self.fixed_point_abs_errors = []
        self.newton_n_iter_list = []
        self.newton_abs_errors = []
        self.n_lu = 0

    def print(self):
        print("-" * 80)
        print("Solver summary:")
        print(f" - solver name: {self.solver_name}")
        if self.fixed_point_n_iter_list:
            print(
                f" - fixed-point iterations: max = {max(self.fixed_point_n_iter_list)}, avg = {sum(self.fixed_point_n_iter_list) / float(len(self.fixed_point_n_iter_list)):.2f}"
            )
            print(
                f" - fixed-point maximal error: max = {max(self.fixed_point_abs_errors):.3e}, avg = {sum(self.fixed_point_abs_errors) / float(len(self.fixed_point_abs_errors)):.3e}"
            )
        if self.newton_n_iter_list:
            print(
                f" - newton iterations: max = {max(self.newton_n_iter_list)}, avg = {sum(self.newton_n_iter_list) / float(len(self.newton_n_iter_list)):.2f}"
            )
            print(
                f" - newton maximal error: max = {max(self.newton_abs_errors):.3e}, avg = {sum(self.newton_abs_errors) / float(len(self.newton_abs_errors)):.3e}"
            )
        if self.n_lu:
            print(f" - performed lu-decompositions: {self.n_lu}")
        print("-" * 80)

    def add_fixed_point(self, n_iterations, abs_error):
        self.fixed_point_n_iter_list.append(n_iterations)
        self.fixed_point_abs_errors.append(abs_error)

    def add_newton(self, n_iterations, abs_error):
        self.newton_n_iter_list.append(n_iterations)
        self.newton_abs_errors.append(abs_error)

    def add_lu(self, n):
        self.n_lu += n
