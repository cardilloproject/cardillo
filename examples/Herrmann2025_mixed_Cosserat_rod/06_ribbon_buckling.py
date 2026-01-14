from math import pi
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from warnings import warn

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.math import e2, e3, A_IB_basic
from cardillo.math.approx_fprime import approx_fprime
from cardillo.rods import (
    CircularCrossSection,
    RectangularCrossSection,
    Simo1986,
    RodMaterialModel,
)
from cardillo.rods.force_line_distributed import Force_line_distributed
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import Newton, SolverOptions
from cardillo.utility.sensor import Sensor, SensorRecords


class Ribbon(RodMaterialModel):
    """To be used in combination with a constrained rod which is inextensible, shear-rigid and bending stiff around ez_B (ey_B if bendable_axis is set to be 2). Following Audoly, Neukirch (2021): https://doi.org/10.1016/j.jmps.2021.104457
    eq. 2.9 ff

    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ration
    a : float
        width (ey_B)
    t : float
        thickness (ez_B)
    """

    def __init__(self, E, nu, a, t):
        assert a > t, "Arguments of a and t are wrong!"
        self.bendable_axis = 1

        self.nu = nu
        self.stiffness_factor = E * t**5 / (a**3 * (12 * (1 - nu**2)) ** 2)
        self.kappa_char = (12 * (1 - nu**2)) ** (-0.5) * t / a**2  # eq. 2.7

        self.phi = self.phi_B1

    ###############
    # phi functions
    ###############
    def phi_master(self, v):
        """Eq. 2.8 and its derivatives"""
        v_abs_half = np.abs(v) / 2
        v_abs_half_v = np.sign(v) / 2
        # v_abs_half_vv = 0.0

        s = np.sqrt(v_abs_half)
        s_v = 0.5 / s * v_abs_half_v
        s_vv = -0.25 / s**3 * v_abs_half_v**2

        # trigonometric
        sinh = np.sinh(s)
        cosh = np.cosh(s)
        sin = np.sin(s)
        cos = np.cos(s)

        num = cosh - cos
        num_v = (sinh + sin) * s_v
        num_vv = (cosh + cos) * s_v**2 + (sinh + sin) * s_vv

        denom = s * (sinh + sin)
        denom_v = s_v * (sinh + sin) + s * (cosh + cos) * s_v
        denom_vv = (
            s_vv * (sinh + sin)
            + 2 * s_v**2 * (cosh + cos)
            + s * (sinh - sin) * s_v**2
            + s * (cosh + cos) * s_vv
        )

        frac = num / denom
        frac_v = (num_v * denom - num * denom_v) / denom**2
        frac_vv = (
            num_vv * denom - num * denom_vv
        ) / denom**2 - 2 * denom_v / denom * frac_v

        phi = 1 / v_abs_half**2 * (0.5 - frac)
        phi_v = (
            -2 / v_abs_half**3 * v_abs_half_v * (0.5 - frac)
            - 1 / v_abs_half**2 * frac_v
        )
        phi_vv = (
            6 / v_abs_half**4 * v_abs_half_v**2 * (0.5 - frac)
            + 2 * 2 / v_abs_half**3 * v_abs_half_v * frac_v
            - 1 / v_abs_half**2 * frac_vv
        )

        return phi, phi_v, phi_vv

    def phi_quartic(self, v):
        phi = 1 / 360 - v**2 / 181_440 + 2_879 * v**4 / 261_534_873_600
        phi_v = -2 * v / 181_440 + 4 * 2_879 * v**3 / 261_534_873_600
        phi_vv = -2 / 181_440 + 3 * 4 * 2_879 * v**2 / 261_534_873_600
        return phi, phi_v, phi_vv

    def phi_inf(self, v):
        v_abs_half = np.abs(v) / 2
        v_abs_half_v = np.sign(v) / 2
        # v_abs_half_vv = 0.0

        s = np.sqrt(v_abs_half)
        s_v = 0.5 / s * v_abs_half_v
        s_vv = -0.25 / s**3 * v_abs_half_v**2

        phi = (s / 2 - 1) / s**5
        phi_v = s_v / 2 / s**5 - 5 * (s / 2 - 1) / s**6 * s_v
        phi_vv = (
            s_vv / 2 / s**5
            - 2 * 5 * s_v**2 / 2 / s**6
            + 6 * 5 * (s / 2 - 1) / s**7 * s_v**2
            - 5 * (s / 2 - 1) / s**6 * s_vv
        )

        return phi, phi_v, phi_vv

    def phi_B1(self, v):
        """Eq. B.1 and its derivatives"""
        if np.abs(v) <= 0.3:
            return self.phi_quartic(v)
        elif np.abs(v) <= 1_800:
            return self.phi_master(v)
        else:
            return self.phi_inf(v)

    ###########
    # potential
    ###########
    def potential(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        """eq. 2.9"""
        dK = B_Kappa - B_Kappa0

        # extract components
        tors = dK[0] / self.kappa_char
        bend = dK[self.bendable_axis] / self.kappa_char
        coupled_inner = self.nu * bend**2 + tors**2

        ####
        # compute individual terms (1/2 inside the bracket)
        ####
        term_bend = 0.5 * bend**2 * (1 - self.nu**2)
        term_tors = tors**2 * (1 - self.nu)
        term_coupled = coupled_inner**2 * self.phi(bend)[0] / 4

        return self.stiffness_factor * (term_bend + term_tors + term_coupled)

    ###################
    # stress resultants
    ###################
    def B_n(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros(3, dtype=float)

    def B_m(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dK = B_Kappa - B_Kappa0

        # extract components
        tors = dK[0] / self.kappa_char
        bend = dK[self.bendable_axis] / self.kappa_char

        tors_kappa = np.zeros(3, dtype=float)
        tors_kappa[0] = 1 / self.kappa_char

        bend_kappa = np.zeros(3, dtype=float)
        bend_kappa[self.bendable_axis] = 1 / self.kappa_char

        coupled_inner = self.nu * bend**2 + tors**2
        coupled_inner_kappa = 2 * self.nu * bend * bend_kappa + 2 * tors * tors_kappa

        ####
        # compute individual terms
        ####
        # term_bend = 0.5 * bend**2 * (1 - self.nu**2)
        term_bend_kappa = bend * bend_kappa * (1 - self.nu**2)

        # term_tors = tors**2 * (1 - self.nu)
        term_tors_kappa = 2 * tors * tors_kappa * (1 - self.nu)

        phi, phi_bend, phi_bend_bend = self.phi(bend)
        phi_kappa = phi_bend * bend_kappa

        # term_coupled = coupled_inner**2 * phi / 4
        term_coupled_kappa = (
            coupled_inner * phi * coupled_inner_kappa / 2
            + coupled_inner**2 * phi_kappa / 4
        )

        B_m = self.stiffness_factor * (
            term_bend_kappa + term_tors_kappa + term_coupled_kappa
        )
        return B_m

    #############
    # stiffnesses
    #############
    def B_n_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def B_n_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dK = B_Kappa - B_Kappa0

        # extract components
        tors = dK[0] / self.kappa_char
        bend = dK[self.bendable_axis] / self.kappa_char

        tors_kappa = np.zeros(3, dtype=float)
        tors_kappa[0] = 1 / self.kappa_char
        # tors_kappa_kappa = np.zeros((3, 3), dtype=float)

        bend_kappa = np.zeros(3, dtype=float)
        bend_kappa[self.bendable_axis] = 1 / self.kappa_char
        # bend_kappa_kappa = np.zeros((3, 3), dtype=float)

        coupled_inner = self.nu * bend**2 + tors**2
        coupled_inner_kappa = 2 * self.nu * bend * bend_kappa + 2 * tors * tors_kappa
        coupled_inner_kappa_kappa = 2 * self.nu * np.outer(
            bend_kappa, bend_kappa
        ) + 2 * np.outer(tors_kappa, tors_kappa)

        ####
        # compute individual terms
        ####
        # term_bend = 0.5 * bend**2 * (1 - self.nu**2)
        # term_bend_kappa = bend * bend_kappa * (1 - self.nu**2)
        term_bend_kappa_kappa = np.outer(bend_kappa, bend_kappa) * (1 - self.nu**2)

        # term_tors = tors**2 * (1 - self.nu)
        # term_tors_kappa = 2 * tors * tors_kappa * (1 - self.nu)
        term_tors_kappa_kappa = 2 * np.outer(tors_kappa, tors_kappa) * (1 - self.nu)

        phi, phi_bend, phi_bend_bend = self.phi(bend)
        phi_kappa = phi_bend * bend_kappa
        phi_kappa_kappa = phi_bend_bend * np.outer(bend_kappa, bend_kappa)

        # term_coupled = coupled_inner**2 * phi / 4
        # term_coupled_kappa = coupled_inner * phi * coupled_inner_kappa / 2 + coupled_inner**2 * phi_kappa / 4 # 2 cancelled  in the first term (1/2 remains), second is now / 4
        term_coupled_kappa_kappa = (
            phi * np.outer(coupled_inner_kappa, coupled_inner_kappa) / 2
            + coupled_inner * np.outer(phi_kappa, coupled_inner_kappa) / 2
            + phi * coupled_inner * coupled_inner_kappa_kappa / 2
            + 2 * coupled_inner * np.outer(coupled_inner_kappa, phi_kappa) / 4
            + coupled_inner**2 * phi_kappa_kappa / 4
        )

        # B_m_ana = self.stiffness_factor * (
        #     term_bend_kappa + term_tors_kappa + term_coupled_kappa
        # )
        B_m_kappa = self.stiffness_factor * (
            term_bend_kappa_kappa + term_tors_kappa_kappa + term_coupled_kappa_kappa
        )
        return B_m_kappa


class Sadowsky(Ribbon):
    def __init__(self, E, nu, a, t):
        super().__init__(E, nu, a, t)
        self.stiffness_factor_sadowsky = E * a * t**3 / (12 * (1 - nu**2))
        self.bend_switch = 1e-6

    def strain(self, bend, tors):
        tors_kappa = np.zeros(3, dtype=float)
        tors_kappa[0] = 1.0

        bend_kappa = np.zeros(3, dtype=float)
        bend_kappa[self.bendable_axis] = 1.0

        bend_switch = self.bend_switch
        if np.abs(bend) > bend_switch:
            # print(f"Large bending!")
            num_strain = bend**2 + tors**2
            num_strain_kappa = 2 * bend * bend_kappa + 2 * tors * tors_kappa
            num_strain_kappa_kappa = 2 * np.outer(
                bend_kappa, bend_kappa
            ) + 2 * np.outer(tors_kappa, tors_kappa)

            strain = num_strain / bend
            strain_kappa = num_strain_kappa / bend - num_strain * bend_kappa / bend**2
            strain_kappa_kappa = (
                num_strain_kappa_kappa / bend
                - np.outer(num_strain_kappa, bend_kappa) / bend**2
                - np.outer(bend_kappa, num_strain_kappa) / bend**2
                + 2 * num_strain * np.outer(bend_kappa, bend_kappa) / bend**3
            )

        elif True:
            # quadratic approximation
            # print(f"Small bending!")
            a = 1 / (2 * bend_switch) - tors**2 / (2 * bend_switch**3)
            a_kappa = -2 * tors * tors_kappa / (2 * bend_switch**3)
            a_kappa_kappa = -2 * np.outer(tors_kappa, tors_kappa) / (2 * bend_switch**3)

            c = (bend_switch**2 + tors**2) / bend_switch - a * bend_switch**2
            c_kappa = 2 * tors * tors_kappa / bend_switch - a_kappa * bend_switch**2
            c_kappa_kappa = (
                2 * np.outer(tors_kappa, tors_kappa) / bend_switch
                - a_kappa_kappa * bend_switch**2
            )

            strain = a * bend**2 + c
            strain_kappa = a_kappa * bend**2 + 2 * a * bend * bend_kappa + c_kappa
            strain_kappa_kappa = (
                a_kappa_kappa * bend**2
                + 2 * bend * np.outer(a_kappa, bend_kappa)
                + 2 * bend * np.outer(bend_kappa, a_kappa)
                + 2 * a * np.outer(bend_kappa, bend_kappa)
                + c_kappa_kappa
            )

        return strain, strain_kappa, strain_kappa_kappa

    def potential(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        """Eq. (2.12), last line"""
        dK = B_Kappa - B_Kappa0

        # extract components
        tors = dK[0]
        bend = dK[self.bendable_axis]

        strain, _, _ = self.strain(bend, tors)

        return 0.5 * self.stiffness_factor_sadowsky * strain**2

    def B_m(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dK = B_Kappa - B_Kappa0

        # extract components
        tors = dK[0]
        bend = dK[self.bendable_axis]
        strain, strain_kappa, _ = self.strain(bend, tors)

        # potential = 0.5 * self.stiffness_factor_sadowsky * strain**2
        B_m = self.stiffness_factor_sadowsky * strain * strain_kappa
        return B_m

    def B_m_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dK = B_Kappa - B_Kappa0

        # extract components
        tors = dK[0]
        bend = dK[self.bendable_axis]
        strain, strain_kappa, strain_kappa_kappa = self.strain(bend, tors)

        # potential = 0.5 * self.stiffness_factor_sadowsky * strain**2
        # B_m = self.stiffness_factor_sadowsky * strain * strain_kappa
        B_m_kappa = self.stiffness_factor_sadowsky * (
            np.outer(strain_kappa, strain_kappa) + strain * strain_kappa_kappa
        )
        return B_m_kappa


def ribbon(
    Rod,
    constitutive_law=Simo1986,
    *,
    geometry: str = "B",
    nelements: int = 10,
    #
    n_load_steps: int = 10,
    #
    show_plots: bool = False,
    VTK_export: bool = False,
    name: str = "simulation",
    save_csv: bool = False,
):
    # handle name
    plot_name = name.replace("_", " ")
    save_name = name.replace(" ", "_")

    ############
    # parameters
    ############
    # cross section properties
    width_a = 0.05  # [5 cm]
    thickness_t = 0.002  # [2 mm]
    cross_section_rect = RectangularCrossSection(width_a, thickness_t)

    # material properties
    E = 210 * 1e9
    mu = 0.4
    G = E / (2 * (1 + mu))
    I_y = width_a * thickness_t**3 / 12
    I_z = thickness_t * width_a**3 / 12  # constrained
    I_p = width_a * thickness_t**3 / 3
    Ei = np.array([5, 1, 1])  # constrained
    Fi = np.array([G * I_p, E * I_y, E * I_z])

    kappa_char = (12 * (1 - mu**2)) ** (-0.5) * thickness_t / width_a**2

    length_A = 0.0157 / kappa_char
    length_B = 0.315 / kappa_char

    if geometry == "A":
        length = length_A
    elif geometry == "B":
        length = length_B
    else:
        raise NotImplementedError(
            f"'{geometry}' is not valid! Only 'A' and 'B' are supported!"
        )

    if constitutive_law == Simo1986:
        material_model = constitutive_law(Ei, Fi)
        sdw_flag = False
    elif constitutive_law == Ribbon:
        material_model = Ribbon(E, mu, width_a, thickness_t)
        sdw_flag = False
    elif constitutive_law == Sadowsky:
        material_model = Sadowsky(E, mu, width_a, thickness_t)
        sdw_flag = True

    # initialize system
    system = System()

    #####
    # rod
    #####
    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(nelements, length, A_IB0=A_IB_basic(np.pi / 2).x)
    # construct cantilever
    cantilever = Rod(
        cross_section_rect,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
    )

    ##########
    # clamping
    ##########
    clamping = RigidConnection(system.origin, cantilever, xi2=0)
    system.add(cantilever, clamping)

    ##################
    # distributed load
    ##################
    gamma_crit_rib = 18.178 / np.sqrt(1 + mu)
    gamma_crit_sdw = 21.491 / (1 - mu**2)
    gamma_bar_Box = (1 - 1e-3) * (gamma_crit_sdw if sdw_flag else gamma_crit_rib)

    tau_bar = 15 / 16
    alpha = 6

    def gamma(tau):
        if tau <= tau_bar:
            return 4 * gamma_crit_rib + (gamma_bar_Box - 4 * gamma_crit_rib) * (
                np.exp(-alpha * tau) - 1
            ) / (np.exp(-alpha * tau_bar) - 1)
        else:
            return gamma_bar_Box * (tau - 1) / (tau_bar - 1)

    factor = E * width_a * thickness_t**3 / (12 * length**3)

    def f_new(t, xi):
        if t <= 0.2:
            direction = e3 - np.pi / 40 * (1 - 5 * t) * e2
            return -factor * (5 * t) * (4 * gamma_crit_rib) * direction
        else:
            tau = (5 * t - 1) / 4
            return -factor * gamma(tau) * e3

    f_grav = Force_line_distributed(f_new, cantilever)
    system.add(f_grav)

    ########
    # Sensor
    ########
    sensor = Sensor(cantilever, xi=1.0, name=f"{save_name}_tip")
    system.add(sensor)

    ##############
    # perturbation
    ##############
    # assemble system
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    ############
    # simulation
    ############
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=50, newton_atol=1e-11),  # rtol=0
    )
    sol = solver.solve()

    #################
    # post-processing
    #################
    # export
    dir_name = Path(__file__).parent

    ry = (
        np.array(
            [sensor._r_OP(ti, qi[sensor.qDOF])[1] for (ti, qi) in zip(sol.t, sol.q)]
        )
        / length
    )
    forces = np.array([f_grav.force(ti, 0.0) for ti in sol.t])
    gammas = np.abs(forces[:, 2]) / gamma_crit_rib / factor

    if show_plots:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f"{plot_name}")
        ax.plot(gammas, ry, "-" if geometry == "B" else "--")
        ax.grid()
        plt.show()

    if save_csv:
        sensor.save(dir_name, f"csv", sol, [SensorRecords.r_OP], plot=False)
        np.savetxt(
            dir_name / f"csv/{save_name}_forces.csv",
            np.column_stack((sol.t, forces)),
            delimiter=", ",
            header="t, F_x, F_y, F_z",
            comments="",
        )

    if VTK_export:
        # add rods and export
        system.export(dir_name, f"vtk/{save_name}", sol)

        from cardillo.solver import Solution

        idx = sol.t >= 0.2
        gamma_norm = -forces[idx, 2][::-1] / gamma_crit_rib / factor
        sol_trim = Solution(system=system, t=gamma_norm, q=sol.q[idx][::-1])

        snapshots = [0, 50, 60, 70, 80]
        t_s = sol.t[idx][::-1][snapshots]
        print(f"t: {t_s}, tau: {(5 * t_s - 1)/4}, gamma_norm: {gamma_norm[snapshots]}")
        system.export(dir_name, f"vtk/{save_name}_trim", sol_trim)

    # check the bending in sadowsky material
    if sdw_flag:
        t_sdw_approx = []
        for i, (ti, qi, la_ci, la_gi) in enumerate(
            zip(sol.t, sol.q, sol.la_c, sol.la_g)
        ):
            for el in range(cantilever.nelement):
                la_ge = la_gi[cantilever.elDOF_la_g[el]]
                for qpi in cantilever.qp[el]:
                    eps_ga, eps_ka = cantilever.eval_strains(
                        ti, qi, None, la_ge, qpi, el
                    )
                    eps_b = eps_ka[material_model.bendable_axis]
                    if np.abs(eps_b) <= material_model.bend_switch:
                        if not ti in t_sdw_approx and ti != 0.0:
                            t_sdw_approx.append(ti)

        # print times, where we are below the bending switch, i.e., the approximation is used
        print(f"Quadratic approximation is used for times: {t_sdw_approx}")
        t_sdw_approx = np.array(t_sdw_approx)
        for ti in t_sdw_approx:
            ti_bar = (5 * ti - 1) / 4

            gammai = gamma(ti_bar)
            gamma_norm = gammai / gamma_crit_rib
            gamma_sdw_norm = gammai / gamma_crit_sdw - 1
            print(
                f"t: {ti:.2f}, ti_bar: {ti_bar:.3f}, gammai: {gammai:.2e}, gamma_norm: {gamma_norm:.3f}, gamma_sdw_norm-1: {gamma_sdw_norm:.3e} (should be less than  1)"
            )


constitutive_laws = {
    Ribbon: [False, "ribbon"],
    Simo1986: [True, "quadratic"],
    Sadowsky: [False, "sadowsky"],
}
if __name__ == "__main__":
    # choose constitutive law
    c_law = Ribbon
    # c_law = Simo1986
    c_law = Sadowsky

    # choose geometry
    geometry = "A"
    # geometry = "B"

    # run simulation
    name = f"W_{constitutive_laws[c_law][1]}_geometry_{geometry}"
    print(f"\n   {name}\n")
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=constitutive_laws[c_law][0],
        polynomial_degree=2,
        reduced_integration=True,
        constraints=[0, 1, 2, 5],
    )
    ribbon(
        Rod,
        constitutive_law=c_law,
        geometry=geometry,
        nelements=32,
        n_load_steps=100,
        show_plots=True,
        name=name,
    )
