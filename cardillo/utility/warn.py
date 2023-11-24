from warnings import warn


def warn_extraneous(extraneous):
    """Display a warning for extraneous keyword arguments.

    Parameters
    ----------
    extraneous : dict
        Extraneous keyword arguments
    """
    if extraneous:
        warn(
            "The following arguments have no effect: {}.".format(
                ", ".join(f"`{x}`" for x in extraneous)
            )
        )
