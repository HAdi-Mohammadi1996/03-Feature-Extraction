# Microstructure Feature Extraction

A Julia workflow for extracting quantitative features from segmented 2D or 3D microstructures stored in MATLAB `.mat` files. It calculates phase volume fractions, specific surface areas, triple-phase-boundary density, and physical tortuosity. Results are written to CSV.

## Repository layout

- `main.jl` — batch-processing entry point.
- `src/` — input/output, connectivity, volume-fraction, surface-area, TPB, and tortuosity functions.
- `inputs/` — input `.mat` files.
- `output/` — generated feature tables.
- `test/` — analytical validation and comparison scripts.
- `config.toml` — input, spacing, direction, smoothing, and AMGX library settings.
- `amgx.json` — AMGX solver configuration.
- `Project.toml` and `Manifest.toml` — Julia environment and pinned dependencies.

## Setup

Use Julia 1.12.x; the committed manifest was generated with Julia 1.12.7. Instantiate the environment from the repository root:

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Tortuosity calculations require a compatible NVIDIA GPU, CUDA installation, and NVIDIA AMGX shared library. Update `AMGX_DLL` in `config.toml` to the local library path.

## Configuration

`config.toml` controls the input directory, MATLAB variable key, transport direction, voxel spacings, and surface-area smoothing. By default, the workflow reads `inputs/*.mat`, loads the array named `C`, and assumes phase labels `1`, `2`, and `3`.

`amgx.json` configures a PCG solver with an aggregation-AMG preconditioner, a relative L2 tolerance of `1e-6`, and a maximum of 500 iterations.

## Run

From the repository root:

```powershell
julia --project=. main.jl
```

With the default configuration, results are written to `output/inputs.csv`.

The current tests are validation scripts rather than a `Pkg.test()` suite. Run them individually from the repository root:

```powershell
julia --project=. test/surface_area_validation.jl
julia --project=. test/tpb_validation.jl
julia --project=. test/physical_tau_validation.jl
```

The optional `test/taufactor_benchmark.py` compares tortuosity against TauFactor and requires Python with NumPy, SciPy, and TauFactor.
