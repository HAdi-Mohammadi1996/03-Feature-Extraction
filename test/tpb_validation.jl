include(joinpath(@__DIR__, "..", "src", "MicrostructureAnalysis.jl"))
import .MicrostructureAnalysis as MA

using GLMakie
using LaTeXStrings

const Nx, Ny, Nz = 20, 20, 20
const PHASE_COLORS = [:gray, :blue, :green]
ticks_l = ([-10, -5, 0, 5, 10], [L"-10", L"-5", L"0", L"5", L"10"])

function repeated_z_edge_tpb_geometry()
    C = zeros(UInt8, Nx, Ny, Nz)
    for k in 1:Nz
        C[1:Nx÷2, 1:Ny÷2, k] .= 1
        C[Nx÷2:Nx, 1:Ny÷2, k] .= 2
        C[1:Nx÷2, Ny÷2:Ny, k] .= 3
        C[Nx÷2:Nx, Ny÷2:Ny, k] .= 3
    end
    return C
end

function main()
    C = repeated_z_edge_tpb_geometry()
        fig = Figure(size=(600, 600))

    ax = Axis3(fig[1,1], aspect=:data, width=Relative(0.9), height=Relative(0.9), xlabel="", ylabel="", zlabel="",
                xticks=ticks_l, yticks=ticks_l, zticks=ticks_l)

    voxels!(ax, -10..10, -10..10, -10..10, C; gap=0.1, color=PHASE_COLORS)

    tpb_calculated = MA.total_tpb_density(C)
    tpb_analytical = 1/(Nx*Ny)
    
    Label(fig[1,1], L"""
    TPB_{\mathrm{analytical}} = %$(round(tpb_analytical, digits=4))
    
    TPB_{\mathrm{numerical}} = %$(round(tpb_calculated, digits=4))
    """;
    halign=:center, valign=:top, tellwidth=false, tellheight=false, fontsize=17)
    
    fig
end

main()
