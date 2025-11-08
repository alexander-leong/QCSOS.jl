using ConicSolve

include("../popt.jl")

function QuantumUnitaryFixedTimeProblem(A, x)
    tr_gram_A = []
    for i in 1:size(A, 1)
        a = real.(A[i, i]' * A[i, i])
        push!(tr_gram_A, a)
    end

    # construct ConeQP problem matrices
    cones::Vector{Cone} = []
    num_blocks = 3
    n = 4
    As = []
    b = []
    c = []
    for i in 1:num_blocks
        f = tr_gram_A[i]
        # A_i, Mπs, b_i = decompose(f, x)
        A, wedderburn = decompose(f, n, x)
        return A, wedderburn
        As = push!(As, A_i)
        b = push!(b, b_i)
        num_vars = Int(sum([size(x, 1) * (size(x, 1) + 1)/2 for x in Mπs]))
        
        # The vector "c_i" is [0 ... 0 -1]
        c_i = zeros(num_vars + 2)
        c_i = set_objective(c_i)
        c = push!(c, c_i)
            
        for Mπs_i in Mπs
            p = size(Mπs_i, 1)
            push!(cones, PSDCone(p))
        end
        push!(cones, NonNegativeOrthant(2))
    end

    # A has the following block structure, this is wrong! just stack the As
    #  As[1]  0 ... 0 0 ... 0
    # 0 ... 0  As[2]  0 ... 0
    # 0 ... 0 0 ... 0  As[3]
    # A = zeros((sum([size(X, 1) for X in As]), size(As[1], 2)))
    # i = 1
    # j = 1
    # for idx in 1:length(As)
    #     num_rows = size(As[idx], 1)
    #     num_cols = size(As[idx], 2)
    #     A[i:num_rows, j:num_cols] = As[idx]
    #     i += num_rows
    #     j += num_cols
    # end

    # constraints on the control coefficients
    # elements in tr_gram_A share the same control coefficients
    num_coeffs = size(As[1], 2)
    for idx in 2:length(As)
        for j in 1:num_coeffs
            A_i = zeros(num_coeffs)
            A_i[j] = 1
            A_i[num_coeffs * idx + j] = -1
            A = vcat(A, A_i)
            b = push!(b, 0)
        end
    end

    num_vars = size(A, 2)
    G = Matrix{Float64}(I, num_vars, num_vars)
    h = zeros(num_vars)
    P = zeros((num_vars, num_vars))

    # Get problem
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp, num_coeffs
end

# Solve problem
function QuantumUnitaryFixedTimeProblem_QSOS(A, x)
    cone_qp, num_coeffs = QuantumUnitaryFixedTimeProblem(A, x)
    kktsolve = "qrchol"
    solver = Solver(cone_qp, kktsolve)
    solver.max_iterations = 10
    status = run_solver(solver)

    min_x = get_solution(solver)
    min_x = x[1:num_coeffs]

    return min_x
end

export QuantumUnitaryFixedTimeProblem