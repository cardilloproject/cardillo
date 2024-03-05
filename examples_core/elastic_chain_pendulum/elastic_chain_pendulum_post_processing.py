from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from cardillo.solver import load_solution

if __name__ == "__main__":
    ###############
    # load solution
    ###############
    path = Path(__file__)
    sol = load_solution(Path(path.parent, "elastic_chain_solution.pkl"))
    system = sol.system
    t = sol.t
    q = sol.q
    u = sol.u

    ############
    # VTK export
    ############
    system.export(path.parent, "vtk", sol)

    ###########################
    # animation with matplotlib
    ###########################

    # export list of partilces
    particles = [
        particle
        for name, particle in system.contributions_map.items()
        if "mass" in name
    ]
    n_particles = len(particles)
    l0 = system.contributions_map["spring_damper_0"].l_ref

    # initialize figure
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    width = 1.5 * n_particles * l0
    ax.axis("equal")
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)

    # prepare data for animation
    frames = len(t)
    target_frames = min(len(t), 200)
    frac = int(frames / target_frames)
    animation_time = 5
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    (line,) = ax.plot([], [], "-ok")

    def update(t, q, line):
        r = np.array([particle.r_OP(t, q[particle.qDOF]) for particle in particles])
        x = [0]
        z = [0]
        x.extend(r[:, 0])
        z.extend(r[:, 2])
        line.set_data(x, z)
        return (line,)

    def animate(i):
        update(t[i], q[i], line)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()
