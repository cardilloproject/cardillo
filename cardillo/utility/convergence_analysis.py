import numpy as np


def __error(t1, t2, f1, f2, measure="lp", kwargs={"p": 1}):
    dt = t1[1] - t1[0]

    def distance_function(x, y):
        """Distance measure for graphs, see Acary2010 and MonteiroMarques1987.
        
        References:
        -----------
        Acary2010: https://doi.org/10.1016/j.apnum.2012.06.026 \\
        MonteiroMarques1987: https://doi.org/10.1016/0022-0396(87)90143-4
        """
        return max(np.abs(x[0] - y[0]), np.linalg.norm(x[1:] - y[1:], ord=2))

    def directed_hausdorff_distance(A, B):
        # from scipy.spatial.distance import directed_hausdorff
        # return directed_hausdorff(A, B)[0]
        return np.max([np.min([distance_function(a, b) for b in B]) for a in A])

    match measure:
        case "uniform":
            # https://en.wikipedia.org/wiki/Uniform_norm
            return np.abs(f1 - f2).max()
        case "lp":
            # https://de.wikipedia.org/wiki/Lp-Raum
            p = kwargs["p"]
            return np.max(np.sum(dt * np.abs(f1 - f2) ** p, axis=0) ** (1 / p))
        case "directed_hausdorff":
            # https://en.wikipedia.org/wiki/Hausdorff_distance
            # https://github.com/mavillan/py-hausdorff/blob/master/hausdorff/hausdorff.py
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
            # https://github.com/scipy/scipy/blob/v1.10.1/scipy/spatial/_hausdorff.pyx
            X1 = np.hstack((t1[:, None], f1))
            X2 = np.hstack((t2[:, None], f2))
            return directed_hausdorff_distance(X2, X1)
        case "hausdorff":
            # https://en.wikipedia.org/wiki/Hausdorff_distance
            # https://github.com/mavillan/py-hausdorff/blob/master/hausdorff/hausdorff.py
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
            # https://github.com/scipy/scipy/blob/v1.10.1/scipy/spatial/_hausdorff.pyx
            X1 = np.hstack((t1[:, None], f1))
            X2 = np.hstack((t2[:, None], f2))
            return max(
                directed_hausdorff_distance(X1, X2), directed_hausdorff_distance(X2, X1)
            )
        case _:
            raise NotImplementedError(f"method '{measure}' is not implemented")


def convergence_analysis(
    get_solver,
    dt_ref,
    final_power,
    power_span,
    states=["q", "u"],
    split_fractions=[],
    atol=1e-12,
    measure="lp",
    visualize=True,
    export=True,
    kwargs={"p": 1},
):
    assert final_power >= max(power_span)
    #####################################
    # compute step sizes with powers of 2
    #####################################
    # simulation time
    t_final = (2**final_power) * dt_ref

    # reference time steps
    t_ref = np.arange(0, t_final + dt_ref, step=dt_ref)
    nt_ref = len(t_ref)

    # powers of the reference time step in descending order
    powers_of_2 = range(max(power_span), min(power_span), -1)

    # corresponding indices in reference time steps
    dt_idx = [np.arange(0, nt_ref, int(2**i)) for i in powers_of_2]

    # extract corresponding subsets from reference time steps
    ts = [t_ref[idx] for idx in dt_idx]

    # generate used time steps
    dts = np.array([dt_ref * 2**i for i in powers_of_2])

    # ensure correctness
    assert t_ref[-1] == t_final
    for i in range(len(ts)):
        assert np.allclose(ts[i][1] - ts[i][0], dts[i])

    # debug information
    print(f"dt_ref: {dt_ref}")
    print(f"dts: {dts}")
    print(f"t_final: {t_final}")

    #############################
    # build error data structures
    #############################
    nsplit = len(split_fractions)
    errors = {}
    for field in states:
        errors[field] = {}
        errors[field]["global"] = []
        if nsplit > 1:
            for j in range(1, nsplit):
                errors[field][f"{split_fractions[j-1]}-{split_fractions[j]}"] = []

    ############################
    # compute reference solution
    ############################
    print(f"compute reference solution:")
    reference = get_solver(t_final, dt_ref, atol).solve()
    print(f"done")

    #########################
    # compute other solutions
    #########################
    for i, dt in enumerate(dts):
        print(f" - i: {i}, dt: {dt:1.1e}")
        sol = get_solver(t_final, dt, atol).solve()

        # comute conforming time grid
        t_ref = reference.t[dt_idx[i]]
        # t_ref = reference.t
        t = sol.t
        assert np.allclose(t_ref, t)

        for field in states:
            ###########
            # 1. global
            ###########
            f_ref = getattr(reference, field)[dt_idx[i]]
            # f_ref = getattr(reference, field)
            f = getattr(sol, field)
            errors[field]["global"].append(__error(t_ref, t, f_ref, f, measure, kwargs))

            ###########
            # 2. splits
            ###########
            if nsplit > 1:
                for j in range(1, nsplit):
                    lower = t.searchsorted(
                        split_fractions[j - 1] * t_final, side="left"
                    )
                    upper = t.searchsorted(split_fractions[j] * t_final, side="left")

                    errors[field][
                        f"{split_fractions[j-1]}-{split_fractions[j]}"
                    ].append(
                        __error(
                            t_ref[lower:upper],
                            t[lower:upper],
                            f_ref[lower:upper],
                            f[lower:upper],
                            measure,
                            kwargs,
                        )
                    )

    if visualize:
        import matplotlib.pyplot as plt

        ##################
        # visualize errors
        ##################
        if nsplit > 1:
            fig, ax = plt.subplots(1, len(split_fractions))
        else:
            fig, ax = plt.subplots(1, 1)
            ax = [ax]

        ax[0].set_title("global")
        ax[0].loglog(dts, dts, "-k", label="dt")
        ax[0].loglog(dts, dts**2, "--k", label="dt^2")
        ax[0].loglog(dts, dts**3, "-.k", label="dt^3")
        ax[0].loglog(dts, dts**4, ":k", label="dt^4")
        ax[0].loglog(dts, dts**5, "-ok", label="dt^5")
        ax[0].loglog(dts, dts**6, "-sk", label="dt^6")
        for field in states:
            ax[0].loglog(dts, errors[field]["global"], label=field, marker="x")
        ax[0].grid()
        ax[0].legend()

        if nsplit > 1:
            for j in range(1, nsplit):
                ax[j].set_title(f"{split_fractions[j-1]}-{split_fractions[j]}")
                ax[j].loglog(dts, dts, "-k", label="dt")
                ax[j].loglog(dts, dts**2, "--k", label="dt^2")
                ax[j].loglog(dts, dts**3, "-.k", label="dt^3")
                ax[j].loglog(dts, dts**4, ":k", label="dt^4")
                ax[j].loglog(dts, dts**5, "-ok", label="dt^5")
                ax[j].loglog(dts, dts**6, "-sk", label="dt^6")
                for field in states:
                    ax[j].loglog(
                        dts,
                        errors[field][f"{split_fractions[j-1]}-{split_fractions[j]}"],
                        label=field,
                        marker="x",
                    )
                ax[j].grid()
                ax[j].legend()

        plt.show()

    if export:
        import sys
        from pathlib import Path

        path = Path(sys.modules["__main__"].__file__)

        header = "dt, dt2, dt3, dt4, dt4, dt6"
        for field in states:
            header += ", " + field

        ###############
        # global errors
        ###############
        export_data = np.vstack((dts, dts**2, dts**3, dts**4, dts**5, dts**6))
        for field in states:
            export_data = np.vstack((export_data, errors[field]["global"]))
        np.savetxt(
            path.parent / f"{path.stem}_global.dat",
            export_data.T,
            delimiter=", ",
            header=header,
            comments="",
        )

        ##############
        # split errors
        ##############
        if nsplit > 1:
            for j in range(1, nsplit):
                export_data = np.vstack((dts, dts**2, dts**3, dts**4, dts**5, dts**6))
                for field in states:
                    export_data = np.vstack(
                        (
                            export_data,
                            errors[field][
                                f"{split_fractions[j-1]}-{split_fractions[j]}"
                            ],
                        )
                    )
                np.savetxt(
                    path.parent
                    / f"{path.stem}_{split_fractions[j-1]}-{split_fractions[j]}.dat",
                    export_data.T,
                    delimiter=", ",
                    header=header,
                    comments="",
                )

    return errors
