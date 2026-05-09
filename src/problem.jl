#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package QCSOS.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using DynamicPolynomials

mutable struct QuantumControlSOSProblem
    H0::Matrix{Float64}
    V::Matrix{Float64}
    T::Float64
    n::Int
    program
    t
    x

    function QuantumControlSOSProblem(H0, V, T, program=ConeQP())
        problem = new()
        problem.H0 = H0
        problem.V = V
        problem.T = T
        problem.n = size(H0, 1) + 1
        problem.program = program
        @polyvar x[1:problem.n]
        problem.x = x
        @polyvar t[1:problem.n]
        problem.t = t
        return problem
    end
end

export QuantumControlSOSProblem