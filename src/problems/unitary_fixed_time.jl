#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package QCSOS.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve

include("../popt.jl")

function QuantumUnitaryFixedTimeProblem(U_target, problem, order=2)
    exp½Ω = est_unitary(problem, order)
    A = exp½Ω' *  U_target - exp½Ω
    f = get_hilbert_schmidt_inner_product(A)
    n = problem.n

    cone_qp = ConeQP()
    program = define_program(cone_qp,
                   minimize(f),
                   f ∈ ConicSolve.SymmetricGroup(n))
    
    build_program(program)
    problem.program = program
    
    return problem
end

export QuantumUnitaryFixedTimeProblem