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
    multipliers::Vector{Float64}
    n::Int
    program::SymmetryReducedConeQP
    t
    x

    function QuantumControlSOSProblem(H0, V, T, n=4)
        problem = new()
        problem.H0 = H0
        problem.V = V
        problem.T = T
        problem.multipliers = Vector{Float64}[]
        problem.n = n
        problem.program = SymmetryReducedConeQP{SymmetricGroupAction}()
        @polyvar x[1:n]
        problem.x = x
        @polyvar t[1:n]
        problem.t = t
        return problem
    end
end

export QuantumControlSOSProblem