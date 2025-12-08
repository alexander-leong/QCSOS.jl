using CUDA
using ConicSolve
using SparseArrays
using SymbolicWedderburn

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
    return A, b
end

function set_objective(total_num_cols)
    # minimize sum(p_i) - t
    c = ones(total_num_cols)
    c[end] = -1
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
    block_summands = Dict()
    num_additional_vars = 1 # FIXME remove?
    total_num_additional_vars = 1
    
    sos_symmetric_groups = [SOS_Symmetric_Group() for _ in 1:num_blocks]
    # num_vars = Int(sum([size(x, 1) * (size(x, 1) + 1)/2 for x in psds]))
    num_vars = zeros(Int, num_blocks)
    Threads.@threads for i in 1:num_blocks
        f = tr_gram_A[i]
        A_i, b_i, n_i, summands, sos_symmetric_group = wedderburn_decompose(f, n, x)
        As[i] = A_i
        bs[i] = b_i
        num_vars[i] = n_i
        block_summands[i] = summands
        sos_symmetric_groups[i] = sos_symmetric_group
    end

    num_rows = [size(A, 1) for A in As]
    num_vars = num_vars[1] # num_vars is the same across blocks, just use the first
    total_num_additional_constraints = 0
    total_num_rows = sum(num_rows) + total_num_additional_constraints
    total_num_cols = num_blocks * num_vars + total_num_additional_vars
    A = zeros((total_num_rows, total_num_cols))
    println("Num rows in A: $(size(A, 1))")
    for i in 1:num_blocks
    println("Length of Equality Constraint Inds $(i): $(length(sos_symmetric_groups[i].equality_constraint_indices))")
    end
	b = zeros(total_num_rows)

    # block constraints
    row_inds = [0, cumsum(num_rows)...]
    for i in 1:num_blocks
        A, b = set_equality_constraints(A, As, b, bs, i, num_blocks, num_vars, row_inds)
        # FIXME update the equality constraint indices j values for each symmetric group
        # to account for offset
        offset = num_vars * (i-1)
        inds = sos_symmetric_groups[i].equality_constraint_indices
        for i in eachindex(inds)
            inds[i] = inds[i] .+ (offset, offset)
        end
    end
    for i in 1:num_blocks-1
        Id = Matrix{Float64}(I, num_vars, num_vars)
        offset = num_vars * i
        A_i = zeros((num_vars, total_num_cols))
        A_i[:, 1:num_vars] = Id
        A_i[:, offset+1:offset+num_vars] = -Id
        A = vcat(A, A_i)
        b = vcat(b, zeros(num_vars))
    end
    # FIXME should check A is full row rank in solver
    println("Number of Vars: $(num_vars)")
    println("Rank of A: $(rank(A))")
    println("Size of A: $(size(A, 1))")
    @assert rank(A) == size(A, 1)
    
    # The vector "c" is [1 ... 1 -1]
    c = set_objective(total_num_cols)
    
    for _ in 1:num_blocks
        for psds_i in block_summands[1] # all summand should have same dims, just use the first
            p = size(psds_i, 1)
            push!(cones, PSDCone(p))
        end
    end
    push!(cones, NonNegativeOrthant(num_additional_vars))
    # FIXME add check cone dims function to solver

    num_vars = size(A, 2)
    G = -Matrix{Float64}(I, num_vars, num_vars)
    P = zeros((num_vars, num_vars))
    h = zeros(num_vars)

    # Get problem
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp, block_summands, sos_symmetric_groups
end

# Solve problem
function QuantumUnitaryFixedTimeProblem_QSOS(A, x)
    cone_qp, block_summands = QuantumUnitaryFixedTimeProblem(A, x)
    kktsolve = "qrchol"
    solver = Solver(cone_qp, kktsolve)
    solver.device = GPU
    solver.max_iterations = 1
    status = run_solver(solver)
    # FIXME check status

    # min_x = get_solution(solver)
    # min_x = x[1:num_coeffs]

    # min_x = get_value_in_original_basis(block_summands, min_x)

    # return min_x
    return
end

export QuantumUnitaryFixedTimeProblem