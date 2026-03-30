using DynamicPolynomials

function ∫(p::AbstractPolynomial, x, x_lower, x_upper)
    # get the indefinite integral
    int_p = antidifferentiate.(p, x)
            
    # get the definite integral
    return subs(int_p, x=>x_upper) - subs(int_p, x=>x_lower)
end

function ∫(M::AbstractMatrix, x, x_lower, x_upper)
    return map(z -> ∫(z, x, x_lower, x_upper), M) 
end

function real_poly(p)
    #=
    Real part of the polynomial
    =#
    sum(
        real(c) * m for (c, m) in zip(coefficients(p), monomials(p))
    )
end

function evaluate_outer_product_monomials(n, t, x)
    v = [p(ones(n)*t) for p in monomials(x, 0:n)]
    return v * v'
end

export evaluate_outer_product_monomials

function square_frobenius_norm(M::AbstractArray)
    #=
    Square of the Frobenius norm of a matrix
    =#
    real_poly(sum(z' * z for z in M))
end

function commutator(a, b)
    a * b - b * a
end 