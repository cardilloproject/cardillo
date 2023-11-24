import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate_beam(
    t, q, beams, scale, scale_di=1, n_r=100, n_frames=10, show=True, repeat=True
):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    # prepare data for animation
    frames = len(q)
    # if frames > 0:
    target_frames = min(frames, 100)
    frac = max(1, int(np.ceil(frames / target_frames)))
    # else:
    #     target_frames = 1
    #     frac = 1
    animation_time = 1
    interval = animation_time * 1000 / target_frames

    t = t[::frac]
    q = q[::frac]

    target_frames = q.shape[0]

    # animated objects
    nodes = []
    center_lines = []
    d1s = []
    d2s = []
    d3s = []
    for beam in beams:
        # beam nodes
        nodes.extend(ax.plot(*beam.nodes(q[0]), "--.b"))

        # beam centerline
        center_lines.extend(ax.plot(*beam.centerline(q[0], num=n_r), "-k"))

        # beam frames
        if n_frames > 0:
            r, d1, d2, d3 = beam.frames(q[0], num=n_frames)
            d1 *= scale_di
            d2 *= scale_di
            d3 *= scale_di
            d1s.append(
                [
                    ax.plot(*np.vstack((r[:, i], r[:, i] + d1[:, i])).T, "-r")[0]
                    for i in range(n_frames)
                ]
            )
            d2s.append(
                [
                    ax.plot(*np.vstack((r[:, i], r[:, i] + d2[:, i])).T, "-g")[0]
                    for i in range(n_frames)
                ]
            )
            d3s.append(
                [
                    ax.plot(*np.vstack((r[:, i], r[:, i] + d3[:, i])).T, "-b")[0]
                    for i in range(n_frames)
                ]
            )

    def update(t, q):
        for i, beam in enumerate(beams):
            # beam nodes
            x, y, z = beam.nodes(q)
            nodes[i].set_data(x, y)
            nodes[i].set_3d_properties(z)

            # beam centerline
            x, y, z = beam.centerline(q, num=n_r)
            center_lines[i].set_data(x, y)
            center_lines[i].set_3d_properties(z)

            # beam frames
            if n_frames > 0:
                r, d1, d2, d3 = beam.frames(q, num=n_frames)
                d1 *= scale_di
                d2 *= scale_di
                d3 *= scale_di
                for j in range(n_frames):
                    x, y, z = np.vstack((r[:, j], r[:, j] + d1[:, j])).T
                    d1s[i][j].set_data(x, y)
                    d1s[i][j].set_3d_properties(z)

                    x, y, z = np.vstack((r[:, j], r[:, j] + d2[:, j])).T
                    d2s[i][j].set_data(x, y)
                    d2s[i][j].set_3d_properties(z)

                    x, y, z = np.vstack((r[:, j], r[:, j] + d3[:, j])).T
                    d3s[i][j].set_data(x, y)
                    d3s[i][j].set_3d_properties(z)

    def animate(i):
        update(t[i], q[i])

    anim = FuncAnimation(
        fig, animate, frames=target_frames, interval=interval, blit=False, repeat=repeat
    )
    if show:
        plt.show()
    return fig, ax, anim
