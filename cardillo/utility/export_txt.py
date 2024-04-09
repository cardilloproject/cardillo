import numpy as np
import sys
from pathlib import Path


def export_txt(
    system,
    solution,
    fields=["q", "u"],
    functions=["g_N"],
    path_append="",
):
    header = "t"
    data = [solution.t[:, None]]

    for field in fields:
        field_value = getattr(solution, field)
        data.append(field_value)
        nf = field_value.shape[1]
        for i in range(nf):
            header += f", {field}{i}"

    for function_name in functions:
        function = getattr(system, function_name)
        narg = function.__code__.co_argcount
        match narg:
            case 2:
                function_value = np.array([function(ti) for ti in solution.t])
            case 3:
                function_value = np.array(
                    [function(ti, qi) for (ti, qi) in zip(solution.t, solution.q)]
                )
            case 4:
                function_value = np.array(
                    [
                        function(ti, qi, ui)
                        for (ti, qi, ui) in zip(solution.t, solution.q, solution.u)
                    ]
                )
            case 5:
                function_value = np.array(
                    [
                        function(ti, qi, ui, u_doti)
                        for (ti, qi, ui, u_doti) in zip(
                            solution.t, solution.q, solution.u, solution.u_dot
                        )
                    ]
                )
            case _:
                raise RuntimeError("We can't have more than four function arguments")

        data.append(function_value)
        nf = function_value.shape[1]
        for i in range(nf):
            header += f", {function_name}{i}"

    data = np.hstack(data)

    path = Path(sys.modules["__main__"].__file__)
    np.savetxt(
        path.parent / (path.stem + path_append + ".dat"),
        data,
        delimiter=", ",
        header=header,
        comments="",
    )
