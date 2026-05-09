#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package QCSOS.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using ConicSolveFR

mutable struct QCSOS_Solver
    p::Int
    problem::QuantumControlSOSProblem
    solver::ConicSolve.Solver
    ϵ::Float64
    η_eps::Float64
    η_lambda::Float64

    function QCSOS_Solver(p::Int, problem::QuantumControlSOSProblem, ϵ::Float64, η_eps::Float64, η_lambda::Float64)
        solver = new()
        solver.p = p
        solver.problem = problem
        solver.ϵ = ϵ
        solver.η_eps = η_eps
        solver.η_lambda = η_lambda
        return solver
    end
end

function solve!(solver::QCSOS_Solver)
    problem = solver.problem
    program = problem.program
    cone_qp = program.cone_qp

    cone_solver = Solver(cone_qp)
    cone_solver.device = CPU
    cone_solver.max_iterations = 4
    cone_solver.tol_optimality = solver.ϵ
    
    x_vec, _ = run_fr_solver(program, cone_solver, true, solver.η_eps, solver.η_lambda)
    return x_vec
end

function get_infidelity(problem, solution)
	# compute infidelity from Hilbert Schmidt inner product (Frobenius norm)^2
	# compute 1 - I_e as in https://qopt.readthedocs.io/en/latest/qopt_features/entanglement_fidelity.html
    Z = evaluate_outer_product_monomials(problem.T, problem.x)
	HS = tr(solution * Z)
    println(diag(solution * Z))
	infidelity = HS / length(problem.H0)
    return infidelity
end

export QCSOS_Solver
export get_infidelity
export solve!