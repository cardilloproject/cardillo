
class SolverSummary():
    def __init__(self):
        self.fixed_point_n_iter_list = []
        self.fixed_point_abs_errors = []
        self.newton_n_iter_list = []
        self.n_lu = 0

    def clear(self):
        self.fixed_point_n_iter_list = []
        self.fixed_point_abs_errors = []
        self.newton_n_iter_list = []
        self.n_lu = 0

    def print(self):
        print("-" * 80)
        print("Solver summary:")
        if self.fixed_point_n_iter_list:
            print(
                f" - fixed-point iterations: max = {max(self.fixed_point_n_iter_list)}, avg = {sum(self.fixed_point_n_iter_list) / float(len(self.fixed_point_n_iter_list)):.2f}"
            )
            print(
                f" - fixed-point maximal absolute error: max = {max(self.fixed_point_abs_errors):.3e}, avg = {sum(self.fixed_point_abs_errors) / float(len(self.fixed_point_abs_errors)):.3e}"
            )
        if self.newton_n_iter_list:
            print(
                f" - newton iterations: max = {max(self.newton_n_iter_list)}, avg = {sum(self.newton_n_iter_list) / float(len(self.newton_n_iter_list)):.2f}"
            )
        if self.n_lu > 0:
            print(f" - performed lu-decompositions: {self.n_lu}")
        print("-" * 80)

    def add_fixed_point(self, n_iterations, abs_error):
        self.fixed_point_n_iter_list.append(n_iterations)
        self.fixed_point_abs_errors.append(abs_error)
    
    def add_newton(self, n_iterations):
        self.newton_n_iter_list.append(n_iterations)

    def add_lu(self, n):
        self.n_lu += n

    
