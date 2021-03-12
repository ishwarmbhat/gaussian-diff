import numpy as np


# Apply a gaussian function to the input
def gaussian(u, kwargs):

    mu = kwargs['mu']
    a = kwargs['a']
    sigma = kwargs['sigma']
    L = kwargs['L']

    # Calculate the center
    u_0 = L / 2
    return a * np.exp(-(u - mu - u_0) ** 2 / (2 * sigma ** 2))


# Heat diffusion equation Forward Euler Scheme
class heat_diffusion:

    def __init__(self, L, T, dx, dt, alpha, init_func, kwargs):
        # Store the parameters
        self.L = L
        self.T = T
        self.dx = dx
        self.dt = dt
        self.alpha = alpha

        # Calculate the update parameter
        self.F = alpha * dt / (dx ** 2)

        # Initialize the grid u
        self.x = np.arange(0, L + dx, dx)
        self.u_1 = init_func(self.x, kwargs)
        self.u = np.zeros(len(self.x))

    # One time update step
    def update(self):
        # Apply diffusion rule for internal points
        Nx = len(self.u)
        for i in range(1, Nx - 1):
            self.u[i] = self.u_1[i] + self.F * (self.u_1[i + 1] - 2 * self.u_1[i] + self.u_1[i - 1])

        # Boundary is set to 0
        self.u[0] = 0
        self.u[Nx - 1] = 0

        # Save the current step as next step
        self.u_1 = self.u

    def diffuse(self):
        # For every time step, call the update step
        for i in range(0, self.T, self.dt):
            self.update()


def main():

    # Diffusion parameters
    Lx = 100
    Tn = 30  # Time limit for diffusion
    Dx = 1  # Grid size
    Dt = 1  # Time step of 1
    ALPHA = 0.2

    gauss_dict = {"mu": 0, "sigma": 25, "a": 1, "L": Lx}

    hd = heat_diffusion(Lx, Tn, Dx, Dt, ALPHA, gaussian, gauss_dict)
    hd.diffuse()


if __name__ == "__main__":
    main()
