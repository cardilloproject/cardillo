from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from warnings import warn


SensorRecords = Enum(
    "Sensor records", ["r_OP", "A_IB", "ex_B", "ey_B", "ez_B", "v_P", "B_Omega"]
)


class Sensor:
    def __init__(
        self,
        subsystem,
        B_r_PQ=np.zeros(3),
        A_BJ=np.eye(3),
        xi=None,
        name="sensor",
    ):
        self.subsystem = subsystem
        self.B_r_PQ = B_r_PQ
        self.A_BJ = A_BJ
        self.xi = xi
        self.name = name

        self.functions = {
            SensorRecords.r_OP: self._r_OP,
            SensorRecords.ex_B: self._ex_B,
            SensorRecords.ey_B: self._ey_B,
            SensorRecords.ez_B: self._ez_B,
            SensorRecords.v_P: self._v_P,
            SensorRecords.B_Omega: self._B_Omega,
        }

    def assembler_callback(self):
        local_qDOF = self.subsystem.local_qDOF_P(self.xi)
        self.qDOF = self.subsystem.qDOF[local_qDOF]

        local_uDOF = self.subsystem.local_uDOF_P(self.xi)
        self.uDOF = self.subsystem.uDOF[local_uDOF]

    def _r_OP(self, t, q):
        return self.subsystem.r_OP(t, q, self.xi, B_r_CP=self.B_r_PQ)

    def _A_IB(self, t, q):
        return self.subsystem.A_IB(t, q, self.xi) @ self.A_BJ

    def _ex_B(self, t, q):
        return self._A_IB(t, q)[:, 0]

    def _ey_B(self, t, q):
        return self._A_IB(t, q)[:, 1]

    def _ez_B(self, t, q):
        return self._A_IB(t, q)[:, 2]

    def _v_P(self, t, q, u):
        return self.subsystem.v_P(t, q, u, self.xi, B_r_CP=self.B_r_PQ)

    def _B_Omega(self, t, q, u):
        return self.A_BJ.T @ self.subsystem.B_Omega(t, q, u, self.xi)

    def save(
        self,
        path,
        folder_name,
        solution,
        functions=[*SensorRecords],
        save=True,
        plot=False,
    ):
        if SensorRecords.A_IB in functions:
            functions.append(SensorRecords.ex_B)
            functions.append(SensorRecords.ey_B)
            functions.append(SensorRecords.ez_B)

            functions.remove(SensorRecords.A_IB)

        # remove duplicates and sort
        functions = sorted(set(functions), key=lambda x: x.value)

        # save and keep order from self.functions
        header = "t"
        names = []
        data = [solution.t]
        for field, fct in self.functions.items():
            if not field in functions:
                continue

            # all vectors are in R^3, but take different amounts of arguments
            narg = fct.__code__.co_argcount
            match narg:
                # self, t, q
                case 3:
                    vec3s = np.array(
                        [
                            fct(ti, qi[self.qDOF])
                            for (ti, qi) in zip(solution.t, solution.q)
                        ]
                    )
                # self, t, q, u
                case 4:
                    vec3s = np.array(
                        [
                            fct(ti, qi[self.qDOF], ui[self.uDOF])
                            for (ti, qi, ui) in zip(solution.t, solution.q, solution.u)
                        ]
                    )

            header += "".join([f", {field.name}_{xyz}" for xyz in "xyz"])
            names.append(field.name)
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

            plt.show()
