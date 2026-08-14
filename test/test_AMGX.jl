import Pkg
Pkg.activate(raw"C:\Users\r43341mm\AMGX_julia")

using SparseArrays
using LinearAlgebra
using CUDA
using AMGX

const AMGX_DLL =
    raw"C:\Users\r43341mm\AMGX\build\Release\amgxsh.dll"

const AMGX_CONFIG = """
{
    "config_version": 2,
    "solver": {
        "solver": "CG",
        "max_iters": 100,
        "tolerance": 1e-12,
        "convergence": "RELATIVE_INI",
        "norm": "L2",
        "monitor_residual": 1,
        "store_res_history": 1,
        "print_solve_stats": 1
    }
}
"""

function amgx_solve(A, b)

    config    = AMGX.Config(AMGX_CONFIG)
    resources = AMGX.Resources(config)

    matrix = AMGX.AMGXMatrix(resources, AMGX.dDDI)
    rhs    = AMGX.AMGXVector(resources, AMGX.dDDI)
    x      = AMGX.AMGXVector(resources, AMGX.dDDI)
    solver = AMGX.Solver(resources, AMGX.dDDI, config)

    try
        # ----------------------------------------------------
        # Convert Julia sparse matrix to ZERO-BASED CSR
        #
        # For this symmetric matrix, Julia CSC columns are
        # equivalent to CSR rows.
        # ----------------------------------------------------

        @assert issymmetric(A)

        row_ptr = Cint.(A.colptr .- 1)
        col_idx = Cint.(A.rowval .- 1)
        values  = Float64.(A.nzval)

        println("\nCSR arrays sent to AMGX:")
        println("row_ptr = ", row_ptr)
        println("col_idx = ", col_idx)
        println("values  = ", values)

        println("\nUploading zero-based CSR directly to AMGX...")

        AMGX.upload!(matrix, row_ptr, col_idx, values)
        AMGX.upload!(rhs, Float64.(b))

        AMGX.set_zero!(x, length(b))

        println("Setting up solver...")
        AMGX.setup!(solver, matrix)

        println("Solving...")
        AMGX.solve!(x, solver, rhs)

        solution = Vector(x)

        println("\nAMGX status     = ", AMGX.get_status(solver))
        println("AMGX iterations = ", AMGX.get_iterations_number(solver))

        return solution

    finally
        close(solver)
        close(x)
        close(rhs)
        close(matrix)
        close(resources)
        close(config)
    end
end


function main()

    println("\n==============================================")
    println("        AMGX ZERO-BASED CSR TEST")
    println("==============================================")

    AMGX.set_libAMGX_path(AMGX_DLL)
    AMGX.initialize()

    try

        A = sparse([
             2.0  -1.0   0.0
            -1.0   2.0  -1.0
             0.0  -1.0   2.0
        ])

        b = [1.0, 0.0, 1.0]

        x_cpu = A \ b

        println("\nCPU solution:")
        println(x_cpu)

        x_amgx = amgx_solve(A, b)

        residual = norm(A * x_amgx - b)

        println("\n==============================================")
        println("RESULT")
        println("==============================================")

        println("CPU solution  = ", x_cpu)
        println("AMGX solution = ", x_amgx)
        println("Residual      = ", residual)

        if all(isfinite, x_amgx) && residual < 1e-10
            println("\n✅ AMGX TEST PASSED")
        else
            println("\n❌ AMGX TEST FAILED")
        end

    finally
        println("\nFinalizing AMGX...")
        AMGX.finalize()
    end
end

main()