#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package QCSOS.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using DynamicPolynomials
using LinearAlgebra
using ConicSolve
using QCSOS
using Random

function get_qc_problem(i = 1)
    T = 0.5
    # Drift Hamiltonian
    H0 = [
        0 0 0;
        0 3.21505101e+10 0;
        0 0 6.23173079e+10
    ];

    H0 ./= norm(H0, Inf);

    # Control Hamiltonian
    V = [
        0 1 0;
        1 0 1.41421356;
        0 1.41421356 0
    ];

    V ./= norm(V, Inf);

    n_samples = 1000
    Random.seed!(6292022)
    problem = QuantumControlSOSProblem(H0, V, T)
    exact_x = -1 .+ 2 * rand(length(problem.x) * n_samples)
    exact_x = reshape(exact_x, (length(problem.x), n_samples))
    U_target = get_unitary(problem, exact_x[:, i])
    return U_target, problem
end

export get_qc_problem

"""
    run_test(ϵ = 1e-6, η_eps = 2e-2, η_lambda = 1e-3, p = 2)

# Parameters:
* `ϵ`: absolute tolerance
* `η_eps`: absolute tolerance to determine exposed face (using duality gap)
* `η_lambda`: absolute tolerance to remove near redundant constraints
* `p`: order of Chebyshev expansion for approximating matrix exponential
"""
function run_test(ϵ = 1e-2, η_eps = 1e-3, η_lambda = 1e-3, p = 2)
    # define optimization problem
    U_target, problem = get_qc_problem()
    problem = QuantumUnitaryFixedTimeProblem(U_target, problem, p)
    @info "Optimization problem constructed successfully."
    
    qcsos_solver = QCSOS_Solver(p, problem, ϵ, η_eps, η_lambda)
    solve!(qcsos_solver)

    # get solution
	solution = get_solution(problem.program)

    # evaluate solution
    coefficients = get_control(problem, solution)
    H_result = get_unitary_from_control(problem, coefficients)
    infidelity = get_infidelity(U_target, H_result)
    @info("Infidelity: $(infidelity)")
    return U_target, H_result, problem, solution
end

U_target, problem, solution = run_test()