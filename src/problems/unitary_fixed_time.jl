#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package QCSOS.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using DynamicPolynomials
using MultivariateBases

const DP = DynamicPolynomials
const MB = MultivariateBases

include("../popt.jl")

function scale_basis_coefficients(::Type{MB.MonomialBasis}, ::Type{MB.ScaledMonomialBasis}, p)
    scaled_coefficients = DP.coefficients(p, MB.ScaledMonomialBasis)
    unscaled_coefficients = DP.coefficients(p, MB.MonomialBasis)
    multipliers = scaled_coefficients ./ unscaled_coefficients
    return scaled_coefficients, multipliers
end

function rescale_polynomial(Basis::Type, f, multipliers)
    unscaled_coefficients = DP.coefficients(f, Basis)
    scaled_coefficients = unscaled_coefficients ./ multipliers
    f = polynomial(scaled_coefficients, monomials(f))
    return f
end

function QuantumUnitaryFixedTimeProblem(U_target, problem, degree=4, order=2)
    exp½Ω = est_unitary(problem, order)
    # A = exp½Ω' *  U_target - exp½Ω
    # f = get_hilbert_schmidt_inner_product(A)
    f = get_infidelity(exp½Ω, U_target)
    scaled_coefficients, multipliers = scale_basis_coefficients(MB.MonomialBasis, MB.ScaledMonomialBasis, f)
    problem.multipliers = multipliers
    f = polynomial(scaled_coefficients, monomials(f))

    cone_qp = ConeQP()
    program = define_program(cone_qp,
                   minimize(f),
                   f ∈ ConicSolve.SymmetricGroup(degree))
    
    build_program(program)
    problem.program = program
    
    return problem
end

export QuantumUnitaryFixedTimeProblem