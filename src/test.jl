# using CairoMakie
using DynamicPolynomials
using LinearAlgebra
using Random

include("degenerate_constraint_excitation_solver.jl")
include("./problems/unitary_fixed_time.jl")

function get_problem_parameters()
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

    n = 4
    @polyvar x[1:n]
    @polyvar t[1:n]
    return H0, V, x, t
end

# function process_result(solver::Solver)
    # println("Reconstructing solution in original basis")
    # x = get_value_in_original_basis(summands, x_vec)

    # plot result
    # create_app(solver)
# end

function run_test()
    T = 0.5
    H0, V, x, t = get_problem_parameters()
    n_samples = 1000
    Random.seed!(6292022)
    exact_x = -1 .+ 2 * rand(length(x) * n_samples)
    exact_x = reshape(exact_x, (length(x), n_samples))
    i = 1
    U_target = get_unitary(H0, T, V, exact_x[:, i])
    exp½Ω = est_unitary(H0, T, V, t, x)
    A = exp½Ω' *  U_target - exp½Ω

    cone_qp, summands, sos_symmetric_group = QuantumUnitaryFixedTimeProblem(A, x)
    println("Optimization problem constructed successfully.")
    println(repeat("-", 144))

    solver = Solver(cone_qp)
    solver = DegenerateConstraintExcitationSolver(cone_qp, sos_symmetric_group)
    solver.ipm_solver.cb_before_iteration = cb_before_iteration
    solver.ipm_solver.device = GPU
    solver.ipm_solver.max_iterations = 4
    x = run_fr_solver(solver.ipm_solver)
    # x_vec = get_solution(solver.ipm_solver)
    # println("Dimension of x_vec: $(length(x_vec))")
    # return solver.ipm_solver
end

f = run_test()
# ipm_solver = run_test()
# process_result(ipm_solver)