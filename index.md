@def title = "Algebraic Reduction and Numerical Optimization Methods for solving Quantum Control Sum of Squares Problems"
@def tags = ["syntax", "code"]
@def author = "Alexander Leong"

# Algebraic Reduction and Numerical Optimization Methods for solving Quantum Control Sum of Squares Problems

\tableofcontents <!-- you can use \toc as well -->

> "As far as the laws of mathematics refer to reality, they are not certain; and as far as they are certain, they do not refer to reality." **Albert Einstein**

This article explores algebraic methods for quantum control polynomial optimization problems. We show how degeneracy, numerical stability and scalability are central issues in quantum control and how ConicSolve.jl has been extended to use face reduction and symmetry reduction (via the Wedderburn decomposition).

Many quantum control problems are posed as polynomial optimization problems. Sum of Squares programming is effective at utilizing symmetric self-dual solvers due to such solvers exhibiting polynomial-time convergence and achieving solutions to high precision. Despite this solving Semidefinite programs and moreover Sum of Squares programs remain challenging. First is the computational blowup in memory and runtime cost as the problem size increases, second is the ill conditioning of solving polynomial optimization problems using the monomial basis, third is the numerical struggle that solvers must deal with.

Sum of squares programming is widely used in control (Lyapunov stability), robotics (trajectory optimization) and the validation and verification of safety critical systems. Global convergence and solution quality are becoming increasingly relevant in quantum control and quantum information sciences. Scaling quantum systems without compromising solution quality remains an ongoing challenge. This motivates the question; how can we adapt existing well established algebraic and computational tools in the quantum realm.

In this article we’ll focus on the fixed-time control problem of finding the control that achieves as close as possible a given target unitary at the end of a given evolution time. The QCPOP method popularized by Bondar et al., 2025 uses TSSOS.jl for solving the sum of squares programming problem. We take a different approach by proposing a computational pipeline based on the ideas discussed in Permenter, 2017 to apply face reduction on top of symmetry reduction as a preprocessing step before solving the SOS problem. This is attractive for several theoretical/practical reasons, it exploits problem structure without relaxing the original problem, these mathematical methods can act as additional layers to extend existing solvers, packages such as SymbolicWedderburn.jl (which we use) already exists for exploiting algebraic structure.

An ongoing challenge with adopting these tools is a robust solver architecture. We’ll show the changes in ConicSolve.jl made to further reduce degeneracy and numerical ill-conditioning, particularly the use of matrix decompositions, equilibration and regularization methods. We hope this article stimulates discussion and interest in further advancing Julia’s ecosystem to realize and explore new algebraic and computational methods. It is without many Julia packages like OperatorScaling.jl and SymbolicWedderburn.jl that this article and exploration of these ideas would remain just that, a theoretical ideal.

## The fixed time control problem

The fixed time control problem involves finding the control coefficients in ``E(t)`` that results in a unitary ``U(T)`` which is as close as possible to a given target unitary, ``U*`` at time ``T``.

Specifically we solve the optimization problem
$$ \begin{aligned}\text{minimize}\qquad & Tr(M^{\dagger}M) \\
\text{subject to}\qquad & \partial_tU(t) = A(t)U(t) \\
\qquad & U(0) = \mathbb{1} \\
\qquad & A(t) = -i(H_0 + E(t)V) \end{aligned} $$

where the objective function in Eq. (1) is approximated as
$$ \|e^{\Lambda_n/2} - e^{-\Lambda_n/2}U^*\|^2 $$

and the control function takes the form
$$ E(t) = \Sigma_{k=0}^{m-1}x_kt^k $$

where $x_k$ are the coefficients being solved for.

```julia
# Drift Hamiltonian
H0 = [
    0 0 0;
    0 3.21505101e+10 0;
    0 0 6.23173079e+10
];

H0 ./= norm(H0, Inf);

# Control Hamiltonian
V = [
    0 1 0;
    1 0 1.41421356;
    0 1.41421356 0
];

V ./= norm(V, Inf);

n = 4
@polyvar x[1:n]
@polyvar t[1:n]

n_samples = 1000
Random.seed!(6292022)
exact_x = -1 .+ 2 * rand(length(x) * n_samples)
exact_x = reshape(exact_x, (length(x), n_samples))
i = 1
U_target = get_unitary(H0, T, V, exact_x[:, i])
exp½Ω = est_unitary(H0, T, V, t, x)
A = exp½Ω' *  U_target - exp½Ω

```

### Estimating the unitary

The Chebyshev polynomial expansion will be used to approximate the matrix exponent
$$ e^{\alpha\Lambda_n}\approx J_0(\alpha)*1 + 2\Sigma_{k=1}^p J_k(\alpha)T_k $$

where \\
- $J_k(x)$ is the Bessel function \\
- $T_k$ is the matrix valued Chebyshev polynomial defined via the recurrence relation $T_{k+1} = 2\Lambda_nT_k + T_{k-1}$ with $T_0 = 1$ and $T_1 = \Lambda_n$

### Sum of Squares Formulation

The quantum control problem requires solving a 4th order polynomial optimization problem.
Expressed using the monomial basis this is:

$v(x) = \left[ 
\begin{array}{ll} 
\text{deg 0:} & 1 \\ \hline
\text{deg 1:} & x_4, x_3, x_2, x_1 \\ \hline
\text{deg 2:} & x_4^2, x_3x_4, x_3^2, x_2x_4, x_2x_3, x_2^2, x_1x_4, x_1x_3, x_1x_2, x_1^2 \\ \hline
\text{deg 3:} & x_4^3, x_3x_4^2, x_3^2x_4, x_3^3, x_2x_4^2, x_2x_3x_4, x_2x_3^2, x_2^2x_4, x_2^2x_3, x_2^3, \\ 
              & x_1x_4^2, x_1x_3x_4, x_1x_3^2, x_1x_2x_4, x_1x_2x_3, x_1x_2^2, x_1^2x_4, x_1^2x_3, x_1^2x_2, x_1^3 \\ \hline
\text{deg 4:} & x_4^4, x_3x_4^3, x_3^2x_4^2, x_3^3x_4, x_3^4, x_2x_4^3, x_2x_3x_4^2, x_2x_3^2x_4, x_2x_3^3, x_2^2x_4^2, \\
              & x_2^2x_3x_4, x_2^2x_3^2, x_2^3x_4, x_2^3x_3, x_2^4, x_1x_4^3, x_1x_3x_4^2, x_1x_3^2x_4, x_1x_3^3, x_1x_2x_4^2, \\
              & x_1x_2x_3x_4, x_1x_2x_3^2, x_1x_2^2x_4, x_1x_2^2x_3, x_1x_2^3, x_1^2x_4^2, x_1^2x_3x_4, x_1^2x_3^2, x_1^2x_2x_4, \\
              & x_1^2x_2x_3, x_1^2x_2^2, x_1^3x_4, x_1^3x_3, x_1^3x_2, x_1^4
\end{array} 
\right] \in \mathbb{R}^{70}$ \\

**Expressing the objective**

Notice that $M^{\dagger}M$ is a matrix of 4th order polynomials.
But we're only minimizing the trace of this matrix which is just a sum of polynomials.
Hence we set the vector ``c`` representing the conic LP objective to the "ones" vector.

```julia
tr_gram_A = []
for i in 1:size(A, 1)
    a = real.(A[i, i]' * A[i, i])
    push!(tr_gram_A, a)
end

n = 4

f = sum(tr_gram_A)

```

**Expressing the constraints**

The constraints $ \partial_tU(t) = A(t)U(t) $ and $ U(0) = \mathbb{1} $ are part of how we represent ``U(T)``, since we evolve under the Schrodinger equation. Hence the code below implicitly enforces this.

```julia
U_target = get_unitary(H0, T, V, exact_x[:, i])
exp½Ω = est_unitary(H0, T, V, t, x)
A = exp½Ω' *  U_target - exp½Ω
```

## Symmetry Reduction

Symmetry reduction is to find a special projection map that takes the problem and maps it onto itself (the fixed point subspace). Symmetry reduction plays a crucial role to reducing the size of the problem and improving numerical conditioning.

### Wedderburn decomposition

For a 4th order polynomial we have 70 monomial terms or $70*71/2$ variables to solve for. ``SymbolicWedderburn.jl`` is a package used by ``ConicSolve.jl`` to obtain a direct sum decomposition of the system to solve for. This results in a block diagonalization of the problem into 4 smaller SDP problems, the largest SDP $13*14/2$ variables to solve for.

The Wedderburn decomposition is used to give a direct-sum representation of the algebra defined by the group action over Symmetric Group 4. This is expressed as \\
$$A = \oplus_{i=1}^rA_i$$ \\
where A is the direct summand data structure returned by ``SymbolicWedderburn.jl``.

### Implementation

The direct summands data structure returned by ``SymbolicWedderburn.jl`` looks like this
~~~<pluto-tree class="Array"><pluto-tree-prefix role="button" tabindex="0"><span class="long"></span><span class="short"></span></pluto-tree-prefix><pluto-tree-items class="Array"><p-r><p-k></p-k><p-v><pre class="no-block"><code>SymbolicWedderburn.DirectSummand{Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}, SymbolicWedderburn.Characters.Character{Cyclotomics.Cyclotomic{Rational{Int64}, SparseArrays.SparseVector{Rational{Int64}, Int64}}, Int64, SymbolicWedderburn.Characters.CharacterTable{PermutationGroups.PermGroup{PermutationGroups.Perm{UInt16}, PermutationGroups.Transversal{UInt16, PermutationGroups.Perm{UInt16}}}, Cyclotomics.Cyclotomic{Rational{Int64}, SparseArrays.SparseVector{Rational{Int64}, Int64}}, PermutationGroups.Orbit{PermutationGroups.Permutation{PermutationGroups.Perm{UInt16}, PermutationGroups.PermGroup{PermutationGroups.Perm{UInt16}, PermutationGroups.Transversal{UInt16, PermutationGroups.Perm{UInt16}}}}}}}}
direct summand of rank 1 with multiplicity 12
12×70 SparseMatrixCSC{Float64, Int64} with 70 stored entries
⎡⠡⠤⢄⢀⠀⡀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠀⠀⠀⠀⠀⠀⎤
⎢⠀⠀⠀⡀⣀⢀⣀⠈⠀⠁⠂⠀⠁⠂⠒⠀⠀⠡⠀⠠⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⎥
⎣⠀⠀⠀⠀⠀⠀⠀⠀⠒⠐⠐⠒⠐⠐⠀⠒⠒⠀⠌⠄⢄⡠⢁⠡⠄⢄⡠⡀⣀⡠⢁⢁⡈⠤⠄⎦
direct summand of rank 1 with multiplicity 5
5×70 SparseMatrixCSC{Float64, Int64} with 48 stored entries
⎡⠀⠀⠀⠁⠉⠈⠉⠀⣀⢀⢀⣀⢀⢀⠀⣀⣀⠀⠠⠀⠀⠀⠄⠄⠀⠀⠀⠀⠀⠀⠄⠄⠠⠀⠀⎤
⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠂⢂⡐⢀⠐⠂⢂⡐⡀⣀⡐⢀⢀⡀⠒⠂⎦
direct summand of rank 1 with multiplicity 13
13×70 SparseMatrixCSC{Float64, Int64} with 89 stored entries
⎡⠈⠉⠑⠐⠀⠂⠀⠢⠀⠄⡀⠀⠄⡀⣀⠀⠀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤
⎢⠀⠀⠀⠂⠒⠐⠒⠀⣀⢀⢀⣀⢀⢀⠀⣀⣀⠈⠠⠈⠀⠀⠄⠄⠈⠀⠀⠀⠀⠀⠄⠄⠠⠀⠈⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠈⠀⠉⠉⠀⠂⠂⢂⡐⢀⠐⠂⢆⡰⡀⣀⡰⢀⢀⡀⠶⠆⎥
⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠁⠉⠁⠈⠈⠁⠀⠀⎦
direct summand of rank 1 with multiplicity 3
3×70 SparseMatrixCSC{Float64, Int64} with 30 stored entries
⎡⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠈⠉⠈⠈⠀⠉⠉⠀⠀⠀⡀⢀⠀⢀⡀⡀⢀⠀⠀⢀⠀⠀⠀⣀⡀⎤
⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⢀⠀⠀⢀⡀⠀⢀⡀⢀⢀⡀⠀⠀⎦
</code></pre></p-v></p-r></pluto-tree-items></pluto-tree>~~~

Unfortunately the returned blocks may be close to (if not) degenerate. Face reduction is a powerful method to remove the structural degeneracy inherent in many quantum control problems.

### A few notes

Reconstructing the solution from the fixed point subspace to obtain the coefficients of the 70 monomial terms is just the sum over each of the direct summand elements and the solution (in matrix form) $X$.
$$ \Sigma_{i=1}^n A_i^TXA_i $$

## Face Reduction

Face reduction is a numerical method that aims to remove structural degeneracy where the solution to the problem lies on a lower dimensional subspace. The following algorithm below implements Algorithm 1.1 as described by Permenter 2017.

$\begin{array}{l}
\textbf{Function} \text{ Face Reduction as SDP}(A) \\
\hspace{1em} \textbf{Inputs: affine set A} \\
\hspace{1em} \textbf{Output: a face F} \\
\hspace{1em} \textbf{while } \textbf{(*) is feasible} \\
\hspace{2em} \textbf{Expose face by solving the following SDP} (*) \\
\hspace{2em}
\begin{aligned}\text{minimize}\qquad & 0 \\
\text{subject to}\qquad & A = \lbrace x \in V : Ax = b \rbrace \end{aligned} \\
\hspace{2em} \textbf{Update face} \\
\hspace{2em} F = US_+^kU^T \\
\hspace{1em} \textbf{Return } F \\
\textbf{End Function}
\end{array}$ \\

$US_+^kU^T$ is found by taking the singular value decomposition of $x$.

The accuracy of the face reduction procedure is dependent on
- the solver's ability to get a near zero duality gap
- a numerical threshold on $\Sigma$ (i.e. $S_+^k$) that is as small as practical

The following optimization problem now solves for the solution in terms of the minimal face

$$\begin{aligned}\text{minimize}\qquad & UCU^Tx \\
\text{subject to}\qquad & A = \lbrace x \in V : Ax = b \rbrace \end{aligned}$$ \\

The solution is then mapped back to the original subspace by reversing the projection operations
$X_n = UX_{n-1}U^T$ which is $X_{n-1} = U^TX_kU$.

## Solving in ConicSolve.jl

ConicSolve.jl is written to solve cone quadratic problems of the form
$$
\begin{aligned}\text{minimize}\qquad & (1/2)x^TPx + c^Tx \\
\text{subject to}\qquad & Gx + s = h \\
\qquad & Ax = b \\
\qquad & s \succeq 0 \end{aligned}
$$

Setting $P = 0$, $c = $ vec$(C)$, $G = -I$ and $h = 0$ gives
$$
\begin{aligned}\text{minimize}\qquad & c^Tx \\
\text{subject to}\qquad & Ax = b \\
\qquad & x \succeq 0
\end{aligned}
$$

We use the optimization problem form (9) throughout this article.

## Dual certificates and optimality conditions

Infeasibility is determined only to within certain numerical tolerances via Farkas-like lemma certificates.
The solver determines a problem is infeasible when the following conditions are met: (i) primal and dual residuals are above a given tolerance, i.e.

Dual residual error
$$
Px + A^Ty + G^Tz + c > \epsilon
$$

Primal residual error
$$
Ax - b > \epsilon
$$

(ii) cone membership of $x$ in cone $K$ fails, i.e. $x \notin K$, and (iii) the norm of $\|y\|$ is large, e.g. > 1e-6

## Results

```julia
julia> ConicSolveFR.run_fr_solver(solver)
[ Info: Performing face reduction on cone 1 of 4
[ Info: Subproblem constructed
[ Info: Iter. |      Dual Res. |    Primal Res. |     Cent. Res. |      Dual. Gap |    Primal Obj. |      Dual Obj. |      Step size |            b'y |
[ Info:     5 |     0.00896941 |    1.13045e-16 |        1.35235 |     0.00361174 |            0.0 |      0.0512984 |       0.137002 |     -0.0512984 |
[ Info: Reduced problem, i = 1
[ Info:     4 |      0.0509707 |    0.000719265 |       0.250474 |      0.0204523 |            0.0 |      0.0351296 |       0.357126 |     -0.0351296 |
[ Info: Reduced problem, i = 2
[ Info:     4 |      0.0109458 |     0.00306507 |      0.0269046 |     0.00554246 |            0.0 |     -0.0065194 |       0.692656 |      0.0065194 |
[ Info: Face reduced from n = 6 to 6
[ Info: Subproblem reduced
[ Info: Iter. |      Dual Res. |    Primal Res. |     Cent. Res. |      Dual. Gap |    Primal Obj. |      Dual Obj. |      Step size |            b'y |
[ Info:     4 |       0.019781 |    0.000643329 |        125.827 |      0.0105947 |      0.0720939 |        1.60794 |      0.0124207 |       -1.60794 |
[ Info: Solved reduced problem
[ Info: Performing face reduction on cone 2 of 4
[ Info: Subproblem constructed
[ Info: Iter. |      Dual Res. |    Primal Res. |     Cent. Res. |      Dual. Gap |    Primal Obj. |      Dual Obj. |      Step size |            b'y |
[ Info:     3 |     0.00174808 |    3.72244e-17 |     0.00209027 |    0.000795396 |            0.0 |    -4.41299e-6 |        0.98973 |     4.41299e-6 |
[ Info: Constraint is weak, largest value 0.0007953878155157315 less than tolerance 0.001, dropping constraint
[ Info: Nothing reduced
[ Info: Performing face reduction on cone 3 of 4
[ Info: Subproblem constructed
[ Info: Iter. |      Dual Res. |    Primal Res. |     Cent. Res. |      Dual. Gap |    Primal Obj. |      Dual Obj. |      Step size |            b'y |
[ Info:     4 |       0.002552 |    5.09224e-17 |      0.0101491 |     0.00116903 |            0.0 |    -0.00594227 |       0.955034 |     0.00594227 |
[ Info: Reduced problem, i = 1
[ Info:     3 |     0.00134631 |     0.00638704 |     0.00648013 |     0.00161668 |            0.0 |    -0.00578691 |       0.943004 |     0.00578691 |
[ Info: Reduced problem, i = 2
[ Info:     3 |    0.000208202 |     0.00883926 |     0.00304389 |    0.000833513 |            0.0 |     -0.0021977 |       0.988035 |      0.0021977 |
[ Info: Reduced problem, i = 3
[ Info:   162 |1.2326000000000002e-32 |      0.0126104 |     3.0686e-20 |    -3.0686e-20 |            0.0 |    3.06741e-20 |   1.11906e-261 |   -3.06741e-20 |
[ Info: Constraint is weak, largest value 0.00035167082990759626 less than tolerance 0.001, dropping constraint
[ Info: Nothing reduced
[ Info: Performing face reduction on cone 4 of 4
[ Info: Subproblem constructed
[ Info: Iter. |      Dual Res. |    Primal Res. |     Cent. Res. |      Dual. Gap |    Primal Obj. |      Dual Obj. |      Step size |            b'y |
[ Info:     3 |    0.000173205 |            0.0 |    0.000173205 |         0.0001 |            0.0 |    2.92253e-12 |           0.99 |   -2.92253e-12 |
[ Info: Constraint is weak, largest value 2.922821715711848e-12 less than tolerance 0.001, dropping constraint
[ Info: Nothing reduced
Face reduction completed
```

## References

* \biblabel{permenter17}{Permenter (2017)} **Permenter**,  Reduction methods in semidefinite and conic optimization, 2017.
* \biblabel{bondar25}{Bondar et al. (2025)} **Bondar**, **Gaggioli**, **Korpas**, **Marecek**, **Vala** and **Jacobs**, [Globally optimal control of quantum dynamics](https://arxiv.org/abs/2209.05790), arXiv:2209.05790v3
* \biblabel{vandenberghe10}{Vandenberghe (2010)} **Vandenberghe**, The CVXOPT linear and quadratic cone program solvers, 2010.
* \biblabel{blekherman13}{Blekherman (2013)} **Blekherman**, **Parrilo** and **Thomas**, Semidefinite Optimization and Convex Algebraic Geometry, 2013.
* \biblabel{leong25}{Leong (2025)} **Leong**, [Efficient Constrained Optimization using ConicSolve.jl](https://github.com/alexander-leong/ConicSolve.jl), 2025.
* \biblabel{kaluba25}{Kaluba et al. (2025)} **Kaluba**, [SymbolicWedderburn.jl](https://github.com/kalmarek/SymbolicWedderburn.jl), 2025.

## Future work

The use of algebraic reduction methods such as symmetry reduction and face reduction are important methods to overcome degeneracy inherent in many quantum control problems. Interior point method solvers such as ConicSolve.jl assume that problems are numerically well conditioned in order for the solver to make progress and obtain a decent solution. Face reduction is numerically challenging and delicate, it requires solving multiple SDP problems and a threshold parameter to identify the minimal face. If the threshold parameter is set too small, the solver would likely stall and become numerically unstable and if set too large information may be lost and the problem being solved diverges from the original problem.

Solving quantum control problems via global optimization methods like Sum of Squares programming remains elusive as quantum systems increasingly make their way into safety critical applications. Future work will likely include exploring alternate basis beyond the monomial basis to further improve numerical stability and robustness and more work is needed to exploit structure so partial face reduction methods can be used to further improve performance.
