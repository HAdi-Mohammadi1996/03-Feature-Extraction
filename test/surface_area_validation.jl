using GLMakie
using LaTeXStrings

include("../src/surface_area.jl")

const Nx, Ny, Nz = 20, 20, 20
ticks_l = ([-10, -5, 0, 5, 10], [L"-10", L"-5", L"0", L"5", L"10"])

function make_cube(;a=Nx÷2)
    C = zeros(UInt8, Nx, Ny, Nz)
    a <= min(Nx, Ny, Nz) || error("large a")
    cx = (Nx + 1) / 2
    cy = (Ny + 1) / 2
    cz = (Nz + 1) / 2

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x = i - cx
        y = j - cy
        z = k - cz

        if abs(x) <= a/2 && abs(y) <= a/2 && abs(z) <= a/2
            C[i, j, k] = 1
        end
    end
    return C
end

function make_cube_rot(;a=Nx÷2, θ=45)
    C = zeros(UInt8, Nx, Ny, Nz)
    a <= min(Nx, Ny, Nz) || error("large a")
    cx = (Nx + 1) / 2
    cy = (Ny + 1) / 2
    cz = (Nz + 1) / 2

    cθ = cosd(θ)
    sθ = sind(θ)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x = i - cx
        y = j - cy
        z = k - cz

        xr = cθ*x + sθ*y
        yr = -sθ*x + cθ*y

        if abs(xr) <= a/2 && abs(yr) <= a/2 && abs(z) <= a/2
            C[i, j, k] = 1
        end
    end
    return C
end

function make_sphere(;r=7)
    C = zeros(UInt8, Nx, Ny, Nz)
    r <= min(Nx, Ny, Nz)÷2 || error("large r")
    O = [Nx÷2, Ny÷2, Nz÷2]

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if (i - O[1])^2 + (j - O[2])^2 + (k - O[3])^2 <= r^2
            C[i, j, k] = 1
        end
    end
    return C
end

function make_multicube()
    C = zeros(UInt8, Nx, Ny, Nz)
    C[1:10, 1:10, 1:5] .= 1
    C[6:15, 6:15, 6:15] .= 2
    C[1:10, 1:10, 16:Nz] .= 1
    C[11:Nx, 6:Ny, 16:Nz] .= 3
    return C
end

const SHAPE_NAMES = [:cube, :cube_rot, :multicube, :sphere]
const PHASE_COLORS = [:gray, :blue, :green]

const GEOMETRIES = Dict(
    :cube => make_cube,
    :cube_rot => make_cube_rot,
    :multicube => make_multicube,
    :sphere => make_sphere
)
const ANALYTICAL_REFERENCE = Dict(
    :cube => 0.075,
    :cube_rot => 0.075,
    :multicube => 0.075,
    :sphere => 49*π/2000
)

function main()

    shape = :sphere

    C = GEOMETRIES[shape]()
    fig = Figure(size=(600, 600))

    ax = Axis3(fig[1,1], aspect=:data, width=Relative(0.9), height=Relative(0.9), xlabel="", ylabel="", zlabel="",
                xticks=ticks_l, yticks=ticks_l, zticks=ticks_l)

    voxels!(ax, -10..10, -10..10, -10..10, C; gap=0.1, color=PHASE_COLORS)

    ssa_calculated = specific_surface_area(C, 1; spacing=(1.0, 1.0, 1.0), σ=0.0)
    ssa_analytical = ANALYTICAL_REFERENCE[shape]
    error_A = 100 * abs(ssa_calculated - ssa_analytical) / ssa_analytical

    Label(fig[1,1], L"""
    a_{s,\mathrm{analytical}} = %$(round(ssa_analytical, digits=4))
    
    a_{s,\mathrm{numerical}} = %$(round(ssa_calculated, digits=4))

    ε = %$(round(error_A, digits=2))\%
    """;
    halign=:center, valign=:top, tellwidth=false, tellheight=false, fontsize=17)
    
    fig
end

main()

# function geom_plot(C, title::String)
#     p = nothing
#     for phase in (1, 2, 3)
#         idx = findall(==(phase), C)
#         isempty(idx) && continue

#         xs = [i.I[1] for i in idx]
#         ys = [i.I[2] for i in idx]
#         zs = [i.I[3] for i in idx]
#         kw = (
#             color=PHASE_COLORS[phase],
#             markershape=:square,
#             markersize=2,
#             markerstrokecolor=:black,
#             markerstrokewidth=0.2,
#             label="",
#         )

#         if p === nothing
#             p = scatter(xs, ys, zs;
#                 kw...,
#                 title=title,
#                 xlim=(0, NX), ylim=(0, NY), zlim=(0, NZ),
#                 aspect_ratio=1,
#                 camera=(45, 30),
#                 legend=false,
#             )
#         else
#             scatter!(p, xs, ys, zs; kw...)
#         end
#     end

#     return p === nothing ? plot(title=title, legend=false) : p
# end

# function save_surface_area_figure(; outdir=joinpath(@__DIR__, "figures"))
#     mkpath(outdir)
#     plots = [voxel_plot(GEOMETRIES[name](), string(name)) for name in SHAPE_NAMES]
#     fig = plot(plots..., layout=(1, 5), size=(1600, 400),
#         plot_title="Surface area test shapes")
#     path = joinpath(outdir, "surface_area_shapes.png")
#     savefig(fig, path)
#     return path
# end

# function validate_surface_area(; make_figure=true)
#     println("="^80)
#     println("Surface Area Validation - Comparison with TauFactor Reference Values")
#     println("="^80)
#     println("Grid resolution: $(NX) x $(NY) x $(NZ) voxels")
#     println("Method: smoothed gradient specific surface area")
#     println("Smoothing parameter (sigma): 1.0")
#     println("Voxel size: 1.0")
#     println("Tolerance: +/- $(100 * SURFACE_AREA_RTOL)%")
#     println("="^80)

#     computed_areas = Dict{Symbol,Float64}()
#     for name in SHAPE_NAMES
#         C = GEOMETRIES[name]()
#         computed_areas[name] = surface_area(C, 1; voxel_size=1.0, sigma=1.0)
#     end

#     figure_path = make_figure ? save_surface_area_figure() : nothing

#     println()
#     println("Validation Results:")
#     println("="^80)
#     println(rpad("Shape", 14), rpad("Computed", 14), rpad("TauFactor", 16),
#         rpad("Diff (%)", 12), "Status")
#     println("-"^80)

#     diffs = Float64[]

#     @testset "TauFactor smoothed-gradient specific surface area" begin
#         for name in SHAPE_NAMES
#             computed = computed_areas[name]
#             reference = TAUFACTOR_REFERENCE[name]
#             diff_pct = 100 * (computed - reference) / reference
#             push!(diffs, abs(diff_pct))

#             passed = isapprox(computed, reference; rtol=SURFACE_AREA_RTOL)
#             status = passed ? "PASS" : "FAIL"

#             @printf("%-14s %.5f        %.5f             %+7.2f %%   %s\n",
#                 String(name), computed, reference, diff_pct, status)

#             @test computed ≈ reference rtol = SURFACE_AREA_RTOL
#         end
#     end

#     println("="^80)
#     println("Mean absolute error: $(round(mean(diffs), digits=2))%")
#     println("Maximum absolute error: $(round(maximum(diffs), digits=2))%")
#     if figure_path !== nothing
#         println("Figure saved to: ", figure_path)
#     end
#     println("="^80)

#     return computed_areas
# end

# results = validate_surface_area()
