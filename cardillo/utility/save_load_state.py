import dill


def save_state(filename):
    dill.dump_session(filename)


def load_state(filename):
    dill.load_session(filename)
