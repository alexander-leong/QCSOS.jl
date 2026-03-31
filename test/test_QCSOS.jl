using DynamicPolynomials
using LinearAlgebra
using ConicSolve
using ConicSolveFR
using QCSOS
using Random

function get_qc_problem()
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

    n = 4
    @polyvar x[1:n]
    @polyvar t[1:n]
    n_samples = 1000
    Random.seed!(6292022)
    exact_x = -1 .+ 2 * rand(length(x) * n_samples)
    exact_x = reshape(exact_x, (length(x), n_samples))
    i = 1
    U_target = get_unitary(H0, T, V, exact_x[:, i])
    return U_target, H0, T, V, t, x
end

export get_qc_problem

function run_test()
    # set QCSOS method parameters
    ϵ = 1e-6 # absolute tolerance
    η_eps = 2e-2 # absolute tolerance to determine exposed face (using duality gap)
    η_lambda = 1e-3 # absolute tolerance to remove near redundant constraints
    p = 2 # order of Chebyshev expansion for approximating matrix exponential

    # define optimization problem
    U_target, H0, T, V, t, x = get_qc_problem()
    exp½Ω = est_unitary(H0, T, V, t, x, p)
    A = exp½Ω' *  U_target - exp½Ω
    cone_qp, summands, _ = QuantumUnitaryFixedTimeProblem(A, x)
    @info "Optimization problem constructed successfully."
    println(repeat("-", 144))

    # solve optimization problem
    solver = Solver(cone_qp)
    solver.device = CPU
    solver.max_iterations = 4
    x_vec, _ = run_fr_solver(solver, true, η_eps, η_lambda)

    # get solution
	solution = get_reduced_solution([summands[1]], [x_vec[1][1]])

	# compute infidelity from Hilbert Schmidt inner product (Frobenius norm)^2
	# compute 1 - I_e as in https://qopt.readthedocs.io/en/latest/qopt_features/entanglement_fidelity.html
    Z = evaluate_outer_product_monomials(length(x), T, x)
	HS = tr(solution * Z)
	infidelity = HS / length(A)
    println("Infidelity: $(infidelity)")
end

f = run_test()