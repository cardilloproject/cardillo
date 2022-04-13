import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_beam(t, q, beam, scale, scale_di=1, n_r=100, n_frames=10, show=True):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    # prepare data for animation
    frames = len(q) - 1
    target_frames = min(frames, 100)
    frac = max(1, int(frames / target_frames))
    animation_time = 1
    interval = animation_time * 1000 / target_frames

    t = t[::frac]
    q = q[::frac]

    # animated objects
    (nodes,) = ax.plot(*beam.nodes(q[0]), "--ob")
    (center_line,) = ax.plot(*beam.centerline(q[0], n=n_r), "-k")
    r, d1, d2, d3 = beam.frames(q[0], n=n_frames)
    d1 *= scale_di
    d2 *= scale_di
    d3 *= scale_di
    global d1s, d2s, d3s
    d1s = [ax.quiver(*r[:, i].T, *d1[:, i].T, color="red") for i in range(n_frames)]
    d2s = [ax.quiver(*r[:, i].T, *d2[:, i].T, color="green") for i in range(n_frames)]
    d3s = [ax.quiver(*r[:, i].T, *d3[:, i].T, color="blue") for i in range(n_frames)]

    def update(t, q):
        # beam nodes
        x, y, z = beam.nodes(q)
        nodes.set_data(x, y)
        nodes.set_3d_properties(z)

        # beam centerline
        x, y, z = beam.centerline(q, n=n_r)
        center_line.set_data(x, y)
        center_line.set_3d_properties(z)

        # animate directors
        global d1s, d2s, d3s
        # # TODO: why is this not working?
        # map(lambda d1: d1.remove(), d1s)
        # map(lambda d2: d2.remove(), d2s)
        # map(lambda d3: d3.remove(), d3s)
        for i in range(n_frames):
            d1s[i].remove()
            d2s[i].remove()
            d3s[i].remove()

        r, d1, d2, d3 = beam.frames(q, n=n_frames)
        d1 *= scale_di
        d2 *= scale_di
        d3 *= scale_di
        d1s = [ax.quiver(*r[:, i].T, *d1[:, i].T, color="red") for i in range(n_frames)]
        d2s = [
            ax.quiver(*r[:, i].T, *d2[:, i].T, color="green") for i in range(n_frames)
        ]
        d3s = [
            ax.quiver(*r[:, i].T, *d3[:, i].T, color="blue") for i in range(n_frames)
        ]

    def animate(i):
        update(t[i], q[i])

    anim = FuncAnimation(
        fig, animate, frames=target_frames, interval=interval, blit=False
    )
    if show:
        plt.show()
    return anim
