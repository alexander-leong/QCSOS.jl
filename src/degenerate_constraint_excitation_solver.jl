using ConicSolve
using ConicSolveFR

mutable struct DegenerateConstraintExcitationSolver
    degenerate_constraints # ordered list of degenerate constraints based on numerical rank
    degenerate_constraints_indices
    ipm_solver
    num_rounds # number of rounds of iterating through degenerate constraints
    sos_symmetric_groups # contains constraint indexing

    function DegenerateConstraintExcitationSolver(program::ConeQP,
                                                  sos_symmetric_groups)
        solver = new()
        solver.degenerate_constraints = []
        solver.degenerate_constraints_indices = []
        solver.ipm_solver = Solver(program)
        solver.num_rounds = 0
        solver.sos_symmetric_groups = sos_symmetric_groups
        return solver
    end
end

function degenerate_psd_predicate(constraint)
    v = constraint.lhs
    N = get_mat_dim(v)
    V = get_mat_from_lt_vec(v, N)
    if cond(V) > 1e3
        return true
    end
    return false
end

function cb_before_iteration(solver::DegenerateConstraintExcitationSolver)
    ipm_solver = solver.ipm_solver
    program = ipm_solver.program
    status = ipm_solver.status
    println("Num rows in A: $(size(program.A, 1))")
    if ipm_solver.current_iteration == 1
        i = 1
        solver.num_rounds += 1
        constraints = find_affine_constraints(program, degenerate_psd_predicate)
        solver.degenerate_constraints = constraints
        # for sos_symmetric_group in solver.sos_symmetric_groups
        # sos_symmetric_group = solver.sos_symmetric_groups
        # for j in sos_symmetric_group.equality_constraint_indices
            # FIXME: remember j is offset for each group
            # v = program.A[i, j[1]:j[2]]
            # N = get_mat_dim(v)
            # V = get_mat_from_lt_vec(v, N)
            # if cond(V) > 1e3
                # push!(solver.degenerate_constraints, v)
                # push!(solver.degenerate_constraints_indices, (i, j[1], j[2]))
            # end
            # i += 1
        # end
        # end
    end
    status_termination = status.status_termination
    # if status_termination == ConicSolve.SLOW_PROGRESS
    if ipm_solver.current_iteration == 1
        println("Relaxing constraints")
        for v in solver.degenerate_constraints
            relax_constraint(solver, v...)
        end
    end
    # elseif status_termination == ConicSolve.OPTIMAL
        # println("Tightening constraints")
        # for (i, v) in enumerate(solver.degenerate_constraints)
            # tighten_constraint(solver, i, v)
        # end
    # end
    # update_affine_constraints(program, ipm_solver)
    println("Condition number of the updated A: $(cond(program.A))")
    println("Optimization problem updated")
end

function cb_after_iteration(solver::DegenerateConstraintExcitationSolver)
end

function regularize(v, tol=1.01e0)
    N = get_mat_dim(v)
    A = get_mat_from_lt_vec(v, N)
    S = svd(A).S
    eps = maximum(S) * tol
    # println("eps: $(eps)")
    A = A + eps * I
    println("Condition number of regularized PSD constraint: $(cond(A))")
    return A, S
end

function relax_constraint(solver::DegenerateConstraintExcitationSolver, i::Int, constraint)
    ipm_solver = solver.ipm_solver
    cone_qp = ipm_solver.program
    lhs = constraint.lhs
    A_i, _ = regularize(lhs)
    a_i = get_vec_from_lt_mat(A_i)
    # idx, j, _ = solver.degenerate_constraints_indices[i]
    # a = get_constraint(cone_qp.A, a_i, j)
    # cone_qp.A[idx, :] = a[:] # moment of truth
    constraint.lhs = a_i
    n = size(cone_qp.A, 2)
    v = zeros(Float64, n)
    cone_qp.A[i, :] = get_constraint(cone_qp, constraint, v)
    # println(cone_qp.A[idx, :] == a[:])
end

function tighten_constraint(solver::DegenerateConstraintExcitationSolver, i::Int, v)
    ipm_solver = solver.ipm_solver
    cone_qp = ipm_solver.program
    a = get_constraint(cone_qp.A, v, i)
    cone_qp.A[i, :] = a
    # TODO: check program is actually being updated!
end

# TODO: refactor solver API
# function ConicSolve.update_solver_status(solver::Solver)
#     r, μ = evaluate_optimality_conditions(solver)

#     # do something with r, μ
#     gap_atol = solver.tol_gap_abs
#     gap_rtol = solver.tol_gap_rel
#     tol = solver.tol_optimality
#     result = is_optimal(solver, gap_atol, gap_rtol, tol)
#     # if result == false
#         # println("WARNING: Solution may not be tight enough")
#     # end

#     return result, r, μ
# end

function tsvd(A, tol=1e-12)
    F = svd(A)
    S::Vector{Float64} = []
    for v in F.S
        if v >= tol
            push!(S, v)
        end
    end
    U = F.U[:, 1:length(S)]
    Vt = F.Vt[1:length(S), :]
    A = U * diagm(S) * Vt
    return Hermitian(A), U, S
end

function reduce_psd(A, U_map, i)
    A, U, S = tsvd(A)
    U_map[i] = (U, S)
    # return diagm(S)
    return A
end

function evaluate_condition(Mπs_j, U_map, i)
    cn = cond(Mπs_j)
    # if cn > 1e2
    Mπs_j = reduce_psd(Mπs_j, U_map, i)
    # println("WARNING: Poorly conditioned PSD condition generated! (condition number = $(cn))")
    # if length(S) == 3 && σ_max > 0 && isapprox(σ_min, 0)
    #     println(Mπs_j)
    # end
    # end
    return Mπs_j
end