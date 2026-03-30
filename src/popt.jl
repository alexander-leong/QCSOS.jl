include("operators.jl")

using DynamicPolynomials
using LinearAlgebra
using QuantumOptics
using SpecialFunctions

"""
Chebyshev approximation for exp(Ω/2)
"""
function exp_chebyshev(Ω::AbstractMatrix, order::Integer)
    
    Tₙ₋₁ = I
    Tₙ  = Ω
    
    # The first two terms of Chebyshev series for exp
    series = besselj(0, 0.5) * Tₙ₋₁ + 2 * besselj(1, 0.5) * Tₙ
    
    for n=2:order
        Tₙ₊₁  = 2 * Ω * Tₙ + Tₙ₋₁
        
        series .+= 2 * besselj(n, 0.5) * Tₙ₊₁
        
        (Tₙ, Tₙ₋₁) = (Tₙ₊₁, Tₙ) 
    end
    
    return series
end

# TODO: generalize the magnus expansion
function est_unitary(H0, T, V, t, x, order=2)
    # get the partial sum of the Magnus expansion
    A₁ = A(H0, V, t[1], x)
    A₂ = A(H0, V, t[2], x)

    Ω = ∫(A₁, t[1], 0, T);

    # 2nd term in the Magnus expansion
    Ω .+= 1//2 * ∫(∫(
        commutator(A₁, A₂), 
        t[2], 0, t[1]), 
        t[1], 0, T
    );

    # 3nd term in the Magnus expansion

    A₃ = A(H0, V, t[3], x)

    Ω .+= 1//6 * ∫(∫(∫(
        commutator(A₁, commutator(A₂, A₃)) + commutator(commutator(A₁, A₂), A₃),
        t[3], 0, t[2]),
        t[2], 0, t[1]),
        t[1], 0, T
    );
    
    Ω = convert(typeof(A₁), Ω)

    exp½Ω = exp_chebyshev(Ω, order);

    return exp½Ω
end

function get_unitary(H0, T, V, x::AbstractArray)
    #=
    Get the unitary given the coefficients for the polynomial control
    =#
    basis = NLevelBasis(size(H0)[1])

    𝓗₀ = DenseOperator(basis, basis, H0)
    𝓥 = DenseOperator(basis, basis, V)

    H = LazySum([1., u(0, x)], [𝓗₀, 𝓥])

    function 𝓗(t, psi)
        H.factors[2] = u(t, x)
        return H
    end

    _, 𝓤 = timeevolution.schroedinger_dynamic([0, T], identityoperator(basis, basis), 𝓗)

    return Matrix(𝓤[2].data)
end

function u(t, x)
    # the polynomial shape for control
    sum(x[n] * t^(n - 1) for n = 1:length(x))
end

function A(H0, V, t, x)
    #=
    The generator of motion entering the Magnus expansion
    =#
    (H0 + V * u(t, x)) / im
end

export est_unitary
export get_unitary