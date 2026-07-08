#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package QCSOS.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using ConicSolveFR
using LinearAlgebra

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
    program_int = program.program_int
    cone_qp = program_int.cone_qp

    cone_solver = Solver(cone_qp)
    cone_solver.device = CPU
    cone_solver.max_iterations = 4
    cone_solver.tol_optimality = solver.ϵ
    
    x_vec, _ = run_fr_solver(program, cone_solver, true, solver.η_eps, solver.η_lambda)
    return x_vec
end

function get_infidelity(solution::Matrix{ComplexF64}, target::Matrix{ComplexF64})
	# compute infidelity from Hilbert Schmidt inner product (Frobenius norm)^2
	# compute 1 - I_e as in https://qopt.readthedocs.io/en/latest/qopt_features/entanglement_fidelity.html
    HS = abs.(tr(target' * solution))^2
	infidelity = HS / length(solution)
    return infidelity
end

function get_control(problem::QuantumControlSOSProblem,
        solution::Matrix{Float64},
        numerical_rank_tol::Float64 = 1e-3)
    polynomial_fn = problem.program.group.f
    
    #= get polynomial control coefficients
       we need to factorize the gram matrix before coefficient read out
       since M' * M == F.Vt' * diagm(F.S.^2) * F.Vt where F = svd(M)
       (see Sanjay Lall's SOS slides, page 9 or similar!) =#
    
    # (i) compute SVD of gram matrix (solution)
    F = svd(solution)
    
    # (ii) get rank one approximation to the gram matrix
    numerical_rank = sum(F.S .> numerical_rank_tol)
    @info("Numerical rank of solution (ideally 1) with tolerance $(numerical_rank_tol) is $(numerical_rank)")
    S = F.S[1]
    Vt = F.Vt[1, :]
    
    # (iii) read out coefficients from S * Vt
    v = S * Vt
    vars = variables(polynomial_fn.f)
    coefficients = [(vars[i], c) for (i, c) in enumerate(reverse(v[2:length(vars)]))]
    return coefficients
end

function get_unitary_from_control(problem::QuantumControlSOSProblem, coefficients)
    # call something like est_unitary, need to check polynomial, u(t, x) first
    H_result = est_unitary(problem.H0, problem.T, problem.V, problem.t, [v[2] for v in coefficients])
    H_result_val = zeros(ComplexF64, size(H_result))
    for i in axes(H_result, 1)
        for j in axes(H_result, 2)
            H_result_val[i, j] = H_result[i, j].a[1]
        end
    end
    return H_result_val
end

export QCSOS_Solver
export get_control
export get_infidelity
export get_unitary_from_control
export solve!