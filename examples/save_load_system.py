import matplotlib.pyplot as plt
from cardillo.utility.save_load_state import load_state

if __name__ == "__main__":
    # load saved state from example "harmonic_oscillator.py"
    load_state("harmonic_oscillator_fcn.pkl")

    print("loaded")
    print(system.contributions)

    plt.plot(t, q[:, 0], "-r")
    plt.plot(t, q[:, 1], "--g")
    plt.plot(t, q[:, 2], "-.b")
    plt.show()
