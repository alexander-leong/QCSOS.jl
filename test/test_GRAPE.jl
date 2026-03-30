using QCSOS

using StaticArrays: SVector

basis = [
    SVector{3}(ComplexF64[1, 0, 0]),
    SVector{3}(ComplexF64[0, 1, 0]),
    SVector{3}(ComplexF64[0, 0, 1])
];

const GHz = 1.0;
const MHz = 0.001GHz;
const ns = 1.0;

using LinearAlgebra: ⋅
δ₁ = 100⋅2π⋅MHz;

using QuantumPropagators.Amplitudes: ShapedAmplitude

T = 400ns;

using QuantumPropagators.Shapes: flattop
shape(t) = flattop(t, T = T, t_rise = 15ns);

ϵ_re_guess(t; Ω₀ = 35⋅2π⋅MHz) = Ω₀
ϵ_im_guess(t) = 0.0

Ω_re_guess = ShapedAmplitude(ϵ_re_guess; shape);
Ω_im_guess = ShapedAmplitude(ϵ_im_guess; shape);

tlist = collect(range(0, T, step = 0.1ns));

import QuantumPropagators: hamiltonian
using StaticArrays: SMatrix

function get_hamiltonian(; δ₁, δ₂, J, Ω_re, Ω_im)

    Ĥ₀ = SMatrix{3,3,ComplexF64}(
        [0 0 0;
        0 3.21505101e+10 0;
        0 0 6.23173079e+10]
    )

    Ĥ₁_re = (1/2) * SMatrix{3,3,ComplexF64}(
        [0 1 0;
        1 0 1.41421356;
        0 1.41421356 0]
    )

    Ĥ₁_im = (𝕚/2) * SMatrix{3,3,ComplexF64}(
        [0 0 0;
        0 0 0;
        0 0 0]
    )

    return hamiltonian(Ĥ₀, (Ĥ₁_re, Ω_re), (Ĥ₁_im, Ω_im))

end

Ĥ = get_hamiltonian(;
    δ₁ = 100⋅2π⋅MHz,
    δ₂ = -100⋅2π⋅MHz,
    J = 3⋅2π⋅MHz,
    Ω_re = Ω_re_guess,
    Ω_im = Ω_im_guess,
)

U_target, _, _, _, _, _ = get_qc_problem()

target_states = [
    SVector{3}(ComplexF64.(U_target[:, 1])),
    SVector{3}(ComplexF64.(U_target[:, 2])),
    SVector{3}(ComplexF64.(U_target[:, 3]))
];

using QuantumPropagators: propagate, Cheby

pops = propagate(
    basis[3], Ĥ, tlist; method=Cheby, storage=true, observables=(Ψ -> Array(abs2.(Ψ)), )
)

using GRAPE: Trajectory

trajectories = [
    Trajectory(Ψ, Ĥ; target_state=Ψ_tgt)
    for (Ψ, Ψ_tgt) in zip(basis, target_states)
]

using QuantumPropagators.Controls: get_controls

get_controls(trajectories)

import Zygote
using GRAPE

GRAPE.set_default_ad_framework(Zygote)

using QuantumControl.Functionals: J_T_re

result = GRAPE.optimize(
    trajectories,
    tlist;
    prop_method = Cheby,
    J_T = J_T_re,
    callback = GRAPE.make_grape_print_iters(),
    iter_stop = 200,
    check_convergence = (res -> ((res.J_T < 1e-2) && "Gate error < 10⁻²")),
    upper_bound = 50⋅2π⋅MHz,
    lower_bound = -50⋅2π⋅MHz,
    use_threads = true,
)

ϵ_opt = result.optimized_controls[1] + 𝕚 * result.optimized_controls[2];