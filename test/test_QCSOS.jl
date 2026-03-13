using DynamicPolynomials
using LinearAlgebra
using Random
using QCSOS

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
    solver.device = GPU
    solver.max_iterations = 4
    x = run_fr_solver(solver)
    # x_vec = get_solution(solver.ipm_solver)
    # println("Dimension of x_vec: $(length(x_vec))")
    # return solver.ipm_solver
end

f = run_test()