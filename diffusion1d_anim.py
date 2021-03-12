import matplotlib.pyplot as plt
from diffusion_1d import heat_diffusion, gaussian
from matplotlib.animation import FuncAnimation
import numpy as np


def main():
    # Diffusion parameters
    Lx = 100
    Tn = 200  # Time limit for diffusion
    Dx = 1  # Grid size
    Dt = 1  # Time step of 1
    ALPHA = 0.75

    gauss_dict = {"mu": 0, "sigma": 25, "a": 1, "L": Lx}

    hd = heat_diffusion(Lx, Tn, Dx, Dt, ALPHA, gaussian, gauss_dict)

    init_lim = np.int(np.max(np.abs(hd.u_1))) + 0.5

    fig = plt.figure()
    ax = plt.axes(xlim=(hd.x.min(), hd.x.max()), ylim=(-init_lim, init_lim))
    line, = ax.plot(hd.x, hd.u_1, lw=3)

    # Initializing animation
    def graph_init():
        line.set_data([], [])
        return line,

    def animate(i, diff_obj):
        diff_obj.update()
        line.set_data(hd.x, hd.u)
        return line,

    anim = FuncAnimation(fig, animate, init_func=graph_init, frames=Tn, fargs=(hd,),
                         interval=20, blit=True)
    anim.save('gauss_diff.gif', writer='imagemagick')


if __name__ == "__main__":
    main()
