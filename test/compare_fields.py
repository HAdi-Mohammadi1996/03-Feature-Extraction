import numpy as np
import matplotlib.pyplot as plt
import taufactor as tau
from scipy.io import loadmat

# --------------------------------------------------
# Settings
# --------------------------------------------------
C_file = "inputs/2.mat"
C_key = "C"

julia_file = "inputs/phi_julia.mat"
julia_key = "phi"

phase = 1
direction = 1       # 1=x, 2=y, 3=z


# --------------------------------------------------
# Load microstructure
# --------------------------------------------------
C = loadmat(C_file)[C_key]

# Conductive phase for TauFactor
img = (C == phase).astype(np.uint8)

# TauFactor solves along axis 0
axis = direction - 1
img_tf = np.moveaxis(img, axis, 0)


# --------------------------------------------------
# TauFactor solve
# --------------------------------------------------
s = tau.Solver(img_tf, device="cuda")
s.solve(conv_crit=1e-5, verbose=False)

print("TauFactor tau  =", s.tau)
print("TauFactor Deff =", s.D_eff)


# --------------------------------------------------
# Extract TauFactor concentration field
# --------------------------------------------------

# Remove batch dimension and one ghost layer
phi_tf = s.field[0, 1:-1, 1:-1, 1:-1].detach().cpu().numpy()

# TauFactor internally converts 2D -> (Nx, Ny, 1)
if C.ndim == 2:
    phi_tf = phi_tf[:, :, 0]

# TauFactor: -0.5 at inlet, +0.5 at outlet
# Julia:      1.0 at inlet,  0.0 at outlet
phi_tf = 0.5 - phi_tf

# Restore original transport-axis orientation
phi_tf = np.moveaxis(phi_tf, 0, axis)


# --------------------------------------------------
# Load Julia field
# --------------------------------------------------
import h5py

with h5py.File(julia_file, "r") as f:
    phi_julia = np.array(f[julia_key])

# MATLAB/HDF5 may reverse dimension order
if phi_julia.shape == C.shape[::-1]:
    phi_julia = np.transpose(phi_julia, axes=range(phi_julia.ndim-1, -1, -1))

print("C shape:", C.shape)
print("Julia phi shape:", phi_julia.shape)

assert phi_julia.shape == C.shape
assert phi_tf.shape == C.shape


# --------------------------------------------------
# Compare only selected phase
# --------------------------------------------------
mask = C == phase

error = phi_julia[mask] - phi_tf[mask]

print("RMSE      =", np.sqrt(np.mean(error**2)))
print("MAE       =", np.mean(np.abs(error)))
print("Max error =", np.max(np.abs(error)))


# --------------------------------------------------
# Plot comparison
# --------------------------------------------------
if C.ndim == 2:

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].imshow(phi_julia.T, origin="lower")
    ax[0].set_title("Julia")

    ax[1].imshow(phi_tf.T, origin="lower")
    ax[1].set_title("TauFactor")

    im = ax[2].imshow(
        (phi_julia - phi_tf).T,
        origin="lower"
    )
    ax[2].set_title("Julia - TauFactor")
    plt.colorbar(im, ax=ax[2])

else:

    # middle z slice
    k = C.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].imshow(phi_julia[:, :, k].T, origin="lower")
    ax[0].set_title("Julia")

    ax[1].imshow(phi_tf[:, :, k].T, origin="lower")
    ax[1].set_title("TauFactor")

    im = ax[2].imshow(
        (phi_julia[:, :, k] - phi_tf[:, :, k]).T,
        origin="lower"
    )
    ax[2].set_title("Julia - TauFactor")
    plt.colorbar(im, ax=ax[2])

plt.tight_layout()
plt.show()