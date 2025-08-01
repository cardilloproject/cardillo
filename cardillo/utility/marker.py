import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from warnings import warn


# TODO: this class fits together with the "rigidly atached rigid body"


class Marker:
    def __init__(
        self,
        subsystem,
        # B_r_PQ=np.zeros(3),
        # A_BJ=np.eye(3),
        xi=None,
        name="marker",
    ):
        self.subsystem = subsystem
        # self.B_r_PQ = B_r_PQ
        # self.A_BJ = A_BJ
        self.xi = xi
        self.name = name

        self.functions = {
            "r_OP": self._r_OP,
            "A_IB": lambda t, q: None,
            "ex_B": lambda t, q: self._A_IB(t, q)[:, 0],
            "ey_B": lambda t, q: self._A_IB(t, q)[:, 1],
            "ez_B": lambda t, q: self._A_IB(t, q)[:, 2],
            "v_P": self._v_P,
            "B_Omega": self._B_Omega,
        }

    def assembler_callback(self):
        local_qDOF = self.subsystem.local_qDOF_P(self.xi)
        self.qDOF = self.subsystem.qDOF[local_qDOF]

        local_uDOF = self.subsystem.local_uDOF_P(self.xi)
        self.uDOF = self.subsystem.uDOF[local_uDOF]

    def _r_OP(self, t, q):
        return self.subsystem.r_OP(t, q, self.xi)

    def _A_IB(self, t, q):
        return self.subsystem.A_IB(t, q, self.xi)

    def _v_P(self, t, q, u):
        return self.subsystem.v_P(t, q, u, self.xi)

    def _B_Omega(self, t, q, u):
        return self.subsystem.B_Omega(t, q, u, self.xi)

    def save(self, path, folder_name, solution, functions, save=True, plot=False):
        if "A_IB" in functions:
            functions.append("ex_B")
            functions.append("ey_B")
            functions.append("ez_B")

            functions.remove("A_IB")

        # remove duplicates
        functions = list(set(functions))

        # check for valid arguments
        invalid = [name for name in functions if name not in self.functions]
        if invalid:
            warn(
                f"Marker '{self.name}': Cannot save: {', '.join(invalid)}", stacklevel=2
            )
            [functions.remove(name) for name in invalid]

        # save and keep order from self.functions
        header = "t"
        names = []
        data = [solution.t]
        for name, fct in self.functions.items():
            if not name in functions:
                continue

            # all vectors are in R^3
            narg = fct.__code__.co_argcount
            match narg:
                case 2 | 3:
                    vec3s = np.array(
                        [
                            fct(ti, qi[self.qDOF])
                            for (ti, qi) in zip(solution.t, solution.q)
                        ]
                    )
                case 4:
                    vec3s = np.array(
                        [
                            fct(ti, qi[self.qDOF], ui[self.uDOF])
                            for (ti, qi, ui) in zip(solution.t, solution.q, solution.u)
                        ]
                    )

            header += "".join([f", {name}_{xyz}" for xyz in "xyz"])
            names.append(name)
            data.extend(vec3s.T)

        # save as csv
        if save:
            # TODO: a bit more structure to directory creation, ...
            # see vtk export
            full_path = Path(path, folder_name)
            full_path.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                full_path / f"{self.name}.csv",
                np.array(data).T,
                delimiter=", ",
                header=header,
                comments="",
            )

        # matplotlib visualization
        if plot:
            fig, ax = plt.subplots(
                3, len(functions), sharex=True, squeeze=False, constrained_layout=True
            )
            for i in range(len(data) - 1):
                ax[i % 3, i // 3].plot(data[0], data[i + 1])

            # add titles, labels and grid
            fig.suptitle(self.name)
            [
                (ax[0, i].set_title(f"{names[i]}"), ax[2, i].set_xlabel("t"))
                for i in range(len(functions))
            ]
            ax[0, 0].set_ylabel("x")
            ax[1, 0].set_ylabel("y")
            ax[2, 0].set_ylabel("z")
            [axii.grid() for axi in ax for axii in axi]

            # plt.tight_layout()
            plt.show()
