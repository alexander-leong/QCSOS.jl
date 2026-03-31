using ConicSolve

include("../popt.jl")

function QuantumUnitaryFixedTimeProblem(A, x)
    tr_gram_A = []
    for i in 1:size(A, 1)
        a = real.(A[i, i]' * A[i, i])
        push!(tr_gram_A, a)
    end

    n = 4
    
    f = sum(tr_gram_A)
    program = ConeQP()
    summands, sos_symmetric_group = wedderburn_decompose(program, f, n, x)
    
    vars = program.vars
    for cone in vars.cones
        add_default_inequality_constraint(program, cone)
        n = ConicSolve.get_size(cone)
        c = ones(n)
        set_objective(program, cone, c)
    end
    
    program = build_program(program)
    
    return program, summands, sos_symmetric_group
end

export QuantumUnitaryFixedTimeProblem