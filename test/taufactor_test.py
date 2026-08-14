import numpy as np
import taufactor as tau
from scipy.io import loadmat
import time


C = loadmat("inputs/2.mat")["C"]

def calculate_tau(C, phase, direction=1):
    """
    C         : 2D or 3D segmented numpy array
    phase     : phase label of interest
    direction : 1, 2, or 3 (same convention as your Julia code)
    """

    # Selected phase = conductive phase
    img = (C == phase).astype(np.uint8)

    # TauFactor solves along axis 0
    axis = direction - 1
    if axis != 0:
        img = np.moveaxis(img, axis, 0)

    # Solve
    s = tau.Solver(img)
    s.solve()

    print("tau   =", s.tau)
    print("D_eff =", s.D_eff)

    return s.tau, s.D_eff

import numpy as np

def tilted_channel(Nx, Ny, h, alpha):
    C = np.zeros((Nx, Ny), dtype=np.uint8)

    m = np.tan(np.deg2rad(alpha))
    yc = Ny / 2

    for i in range(Nx):
        for j in range(Ny):

            yline = yc + m * ((i + 1) - Nx / 2)

            # perpendicular distance to channel centreline
            d = abs((j + 1) - yline) / np.sqrt(1 + m**2)

            if d <= h / 2:
                C[i, j] = 1

    return C


C = tilted_channel(240, 800, 20, 30)
s = tau.Solver(C)
s.solve()

print("TauFactor tau =", s.tau)
print("TauFactor Deff =", s.D_eff)

# tau_value, Deff = calculate_tau(C, phase=1, direction=1)
