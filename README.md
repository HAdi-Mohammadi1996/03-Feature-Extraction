# Microstructure Feature Extraction

This repository contains Julia scripts for extracting morphological and
transport features from 3D labelled microstructures stored in MAT files.
It uses plain scripts rather than a Julia package.

## Input

Place MAT files in the `inputs` directory. Each file must contain a 3D array
named `C` by default. Values greater than zero are treated as material phases;
zero is treated as background.

## Settings

The calculation settings are defined near the top of `main.jl` and
`main_batch.jl`:

```julia
FeatureConfig(
    voxel_size=0.1,
    Nrays=5000,
    min_chord=4.0,
    phases=[1, 2, 3],
    use_all_directions=true,
    direction=1,
    surface_sigma=1.0,
    D0=1.0,
    physical_boundary_factor=2.0,
    physical_rtol=1e-8,
    physical_maxiter=10_000,
    physical_threaded=true,
    physical_thread_threshold=50_000,
    rng_seed=1,
)
```

With `use_all_directions=true`, directional results are averaged over the
three axes. Set it to `false` to use only `direction`.

Physical tortuosity uses non-periodic, zero-flux transverse boundaries and
Dirichlet inlet/outlet boundaries. `physical_boundary_factor=2.0` represents
the half-cell distance between a boundary voxel centre and the boundary face.
The physical tortuosity solver is matrix-free PCG. Start Julia with threads,
for example `julia --threads=8 main.jl`, to use threaded stencil and vector
updates for large systems.

## Run One Sample

The default input is `inputs/2.mat`:

```powershell
julia main.jl
```

An input file and output directory can also be supplied:

```powershell
julia main.jl inputs/3.mat output
```

The result is written to `output/<sample>_features.csv`.

## Run All Samples

Process every `.mat` file in `inputs` and combine the results:

```powershell
julia main_batch.jl
```

Optional input and output directories:

```powershell
julia main_batch.jl inputs output
```

The combined result is written to `output/all_features.csv`.

### Parallel Samples

Samples can be processed concurrently using Julia threads. For two samples at
a time:

```powershell
julia --threads=2 main_batch.jl inputs output 2
```

The third argument is the number of concurrent samples. The default is `1`
because physical tortuosity can use a large amount of memory. Increasing the
worker count helps only when samples are independent and enough RAM is
available for several physical-tortuosity solves at once.

## Output

Each sample is stored in one row. Columns are grouped by property:
`vf1,vf2,vf3`, then `cld1,cld2,cld3`, and so on.

| Column name | Meaning | Unit |
| --- | --- | --- |
| `vf1`, `vf2`, `vf3` | phase volume fraction | dimensionless |
| `cld1`, `cld2`, `cld3` | mean chord length | same length unit as `voxel_size` |
| `sa1`, `sa2`, `sa3` | specific surface area | inverse of the `voxel_size` length unit |
| `gt1`, `gt2`, `gt3` | geometric tortuosity | dimensionless |
| `pt1`, `pt2`, `pt3` | physical tortuosity | dimensionless |
| `perc1`, `perc2`, `perc3` | percolating fraction of each phase | dimensionless |
| `tpb` | total TPB density | inverse square of the `voxel_size` length unit |
| `atpb` | active TPB density | inverse square of the `voxel_size` length unit |

For example, when `voxel_size` is in micrometres, chord length is in
micrometres, specific surface area is in `1/micrometre`, and TPB density is in
`1/micrometre^2`.

TPB calculations use phase labels `1`, `2`, and `3`. Active TPB additionally
requires phases `1` and `2` to percolate in the evaluated direction.

Geometric tortuosity is the mean shortest-path length from every percolating
inlet voxel to the outlet, divided by the straight sample thickness. Paths use
6-neighbour face connectivity.

## Performance Note

Physical tortuosity solves a diffusion system for every requested phase and
direction. The main calculation uses a matrix-free Jacobi-preconditioned
conjugate-gradient solver, avoiding sparse matrix assembly and factorization.
Large volumes such as `256 x 256 x 256` can still require substantial runtime,
especially with `use_all_directions=true`.

For batch runs, avoid oversubscribing the CPU. For large samples, prefer one
sample worker and give the matrix-free solver several Julia threads:

```powershell
julia --threads=8 main_batch.jl inputs output 1
```

When `main_batch.jl` is run with more than one sample worker, the physical
tortuosity solve is kept serial inside each worker to avoid nested threading.

## Dependencies

The scripts require Julia and these external packages:

```julia
import Pkg
Pkg.add(["MAT", "ImageFiltering", "Plots"])
```

`Plots` is only required by the validation scripts.

## Validation

Validation scripts are in `test`:

```powershell
julia test/chord_length_validation.jl
julia test/surface_area_validation.jl
julia test/geometric_tortuosity_validation.jl
julia test/percolation_validation.jl
julia test/physical_tau_validation.jl
julia test/physical_tau_matrixfree_validation.jl
julia test/tpb_validation.jl
```

Generated validation figures are saved in `test/figures`.
