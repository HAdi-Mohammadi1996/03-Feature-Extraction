# Microstructure Feature Extraction

Julia scripts for extracting morphological and transport features from 3D
labelled microstructures stored in MAT files.

## Run-button workflow

No PowerShell arguments are required.

1. Open `main.jl` for one MAT file or `main_batch.jl` for a time series.
2. Edit the settings at the top of that file.
3. Press **Run**.

If the current Julia session has too few threads, the script automatically
restarts itself with `JULIA_COMPUTE_THREADS` and continues the calculation.

## Static phase reuse

The batch configuration contains:

```julia
recalculate_static_phase_each_timestep=false,
static_phase=2,
verify_static_phase=true,
```

- `true` preserves the original behavior and recalculates every phase for every
  MAT file.
- `false` calculates the selected `static_phase` from the first MAT file in
  sorted filename order, then reuses its phase-specific properties at all later
  timesteps.
- `static_phase=2` is the default for YSZ.
- `verify_static_phase=true` checks every later volume against the reference
  phase mask and stops with a clear error if that phase changed.

The reused phase-specific columns are volume fraction, chord length, surface
area, geometric tortuosity, physical tortuosity, and percolating fraction.
`tpb` and `atpb` are always recalculated because they depend on the changing
relationship among phases.

For filenames such as `t000000.mat`, `t000001.mat`, and so on, `t000000.mat`
is the reference because it is the first sorted file.

## Input

Each MAT file must contain a 3D labelled array named `C` by default. Positive
integers are material phases; zero is background. Integer storage types are
preserved, so an `Int32` MAT volume stays `Int32` in memory.

## Main feature settings

```julia
config = FeatureConfig(
    voxel_size=0.1,
    Nrays=5000,
    min_chord=4.0,
    phases=[1, 2, 3],
    use_all_directions=false,
    direction=1,
    surface_sigma=1.0,
    D0=1.0,
    physical_boundary_factor=2.0,
    physical_rtol=1e-8,
    physical_maxiter=10_000,
    physical_threaded=true,
    physical_thread_threshold=50_000,
    physical_max_threads=6,
    recalculate_static_phase_each_timestep=false,
    static_phase=2,
    verify_static_phase=true,
    rng_seed=1,
)
```

With `use_all_directions=true`, directional properties are averaged over all
three axes. Otherwise only `direction` is evaluated.

## Thread settings for this workstation

`main_batch.jl` is configured for the detected AMD Threadripper PRO 7995WX
workstation:

```julia
JULIA_COMPUTE_THREADS = 48
PARALLEL_SAMPLE_WORKERS = 8
PHYSICAL_THREADS_PER_SOLVE = 6
```

Real-data benchmarks on a 256³ sample showed:

| Arrangement | Throughput |
| --- | ---: |
| 1 solve × 48 tasks | about 1.38 solves/minute |
| 3 concurrent solves × 16 tasks | about 3.06 solves/minute |
| 6 concurrent solves × 8 tasks | about 3.35 solves/minute |
| 8 concurrent solves × 6 tasks | about 3.40 solves/minute |
| 12 concurrent solves × 8 tasks on 96 Julia threads | about 3.43 solves/minute |

The 96-thread option provided almost no throughput gain and made a single solve
much slower because of this Windows machine's processor-group/NUMA behavior.
The 48-thread setting is therefore the safer default.

BLAS is restricted to one thread because the matrix-free PCG solver already
uses Julia threads and does not benefit from nested BLAS threading.

## Output

Each sample is one row. Output replacement is atomic, so an interrupted run
does not leave a partly overwritten final CSV.

| Column | Meaning | Unit |
| --- | --- | --- |
| `vf1`, `vf2`, `vf3` | phase volume fraction | dimensionless |
| `cld1`, `cld2`, `cld3` | mean chord length | `voxel_size` length unit |
| `sa1`, `sa2`, `sa3` | specific surface area | inverse length |
| `gt1`, `gt2`, `gt3` | geometric tortuosity | dimensionless |
| `pt1`, `pt2`, `pt3` | physical tortuosity | dimensionless |
| `perc1`, `perc2`, `perc3` | percolating phase fraction | dimensionless |
| `tpb` | total TPB density | inverse length squared |
| `atpb` | active TPB density | inverse length squared |

## Performance changes

- Static-phase properties and percolation masks can be reused.
- Percolation is calculated once per phase/direction and shared by geometric
  tortuosity, physical tortuosity, and active TPB.
- Boundary searches now seed only the relevant 2D face rather than scanning
  the entire 3D volume.
- Linear-index neighbour traversal avoids repeated Cartesian-index conversion.
- The matrix-free PCG solver uses composable dynamic Julia tasks, allowing
  several samples and several solver tasks to share one thread pool.
- MAT integer types are preserved to reduce memory use.

Base Julia threading was retained. `Polyester.jl` is aimed at lower-overhead
thread loops, but the measured workload is dominated by long memory-bound
stencil passes and nested sample/solver scheduling. `LoopVectorization.jl`
targets vectorizable rectangular loops, whereas the main stencil has irregular
neighbour indices and convergence reductions. Neither package offered enough
benefit here to justify an additional dependency or numerical-risk surface.

## Dependencies

The main scripts require:

```julia
MAT
ImageFiltering
```

The validation figure scripts also use `Plots`.

## Validation

Every file in `test` is a directly runnable Julia script. The suite covers:

- chord length,
- surface area,
- percolation,
- geometric tortuosity,
- sparse and matrix-free physical tortuosity,
- TPB,
- full-vs-cached static-phase consistency.

The cache consistency test is `test/feature_cache_validation.jl`.
