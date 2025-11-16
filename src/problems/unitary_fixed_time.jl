using CUDA
using ConicSolve

include("../popt.jl")

function set_equality_constraints(A, As, b, bs, i, num_blocks, num_vars, row_inds)
    # each block may have different number of rows
    row_start_idx = row_inds[i] + 1
    row_end_idx = row_inds[i+1]
    
    # each block must have the same number of columns (variables)
    col_start_idx = num_vars * (i-1) + 1
    col_end_idx = num_vars * i

    # A has the following block structure
    #    As[1]  0 ... 0 0 ... 0
    #   0 ... 0  As[2]  0 ... 0
    #   0 ... 0 0 ... 0  As[3]
    A[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = As[i]
    b[row_start_idx:row_end_idx] = bs[i]

    # set poly - t + delta = 0 s.t. (t, delta) >= 0 and
    # where "t" is the variable (scalar) to optimize
    # There are two additional variables, "t" and "delta":
    #   - x[end-1] is "t"
    #   - x[end] is "delta"
    A[end-i+1, col_start_idx:col_end_idx] .= 1
    # set t_i
    A[end-i+1, end-num_blocks-i+1] = -1
    # set delta_i
    A[end-i+1, end-i+1] = 1
    b[end-i+1] = 0

    # constraints on the control coefficients
    # elements in tr_gram_A share the same control coefficients
    num_coeffs = size(As[1], 2)
    if i >= 2
        for j in 1:num_coeffs
            A_i = zeros((1, size(A, 2)))
            A_i[j] = 1
            A_i[num_coeffs * (i-1) + j] = -1
            A = vcat(A, A_i)
            b = append!(b, 0)
        end
    end
    return A, b
end

function set_objective(num_blocks, total_num_cols)
    c = zeros(total_num_cols)
    c[end-num_blocks+1:end] .= 1
    return c
end

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
    As = [Matrix{Float64}(undef, 0, 0) for _ in 1:num_blocks]
    bs = [Vector{Float64}() for _ in 1:num_blocks]
    c = []
    psds = nothing
    num_additional_vars = 2
    total_num_additional_vars = num_additional_vars * num_blocks
    
    # num_vars = Int(sum([size(x, 1) * (size(x, 1) + 1)/2 for x in psds]))
    num_vars = 0
    Threads.@threads for i in 1:num_blocks
        f = tr_gram_A[i]
        if i == 1
            A_i, b_i, num_vars, psds = decompose(f, n, x)
        else
            A_i, b_i, _, _ = decompose(f, n, x)
        end
        As[i] = A_i
        bs[i] = b_i
    end

    num_rows = [size(A, 1) for A in As]
    total_num_rows = sum(num_rows) + num_blocks
    total_num_cols = num_blocks * num_vars + total_num_additional_vars
    A = zeros((total_num_rows, total_num_cols))
	b = zeros(total_num_rows)

    # block constraints
    row_inds = [0, cumsum(num_rows)...]
    for i in 1:num_blocks
        A, b = set_equality_constraints(A, As, b, bs, i, num_blocks, num_vars, row_inds)
    end
    
    # The vector "c" is [0 ... 0 -1 ... -1]
    c = set_objective(num_blocks, total_num_cols)
    
    for _ in 1:num_blocks
        for psds_i in psds
            p = size(psds_i, 1)
            push!(cones, PSDCone(p))
        end
        push!(cones, NonNegativeOrthant(num_additional_vars))
    end

    num_vars = size(A, 2)
    G = Matrix{Float64}(I, num_vars, num_vars)
    P = zeros((num_vars, num_vars))
    h = zeros(num_vars)

    # Get problem
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp, psds
end

# Solve problem
function QuantumUnitaryFixedTimeProblem_QSOS(A, x)
    cone_qp, psds = QuantumUnitaryFixedTimeProblem(A, x)
    kktsolve = "qrchol"
    solver = Solver(cone_qp, kktsolve)
    solver.device = GPU
    solver.max_iterations = 1
    status = run_solver(solver)

    # min_x = get_solution(solver)
    # min_x = x[1:num_coeffs]

    # min_x = get_value_in_original_basis(psds, min_x)

    # return min_x
    return
end

export QuantumUnitaryFixedTimeProblem