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

    t0 = time.time()
    s = tau.Solver(img)
    s.solve()

    print("tau   =", s.tau)
    print("D_eff =", s.D_eff)
    print("time =", time.time() - t0)

    return s.tau, s.D_eff

calculate_tau(C, phase=2, direction=2)