export RejectionSampler
export InverseTransformSampler
export BoxMullerTransformSampler

@doc raw"""
    RejectionSampler(f, g, proposal_sampler, M)

A rejection sampler.

Rejection sampling involves sampling from a target distribution ``f`` using a
proposal distribution ``g`` which we are able to directly sample from. In order
for the resulting samples to be distributed according to ``f`` we require that
``\frac{f(x)}{g(x)}`` is bounded from above by some constant ``M`` (`M`) for all ``x`` in
the support of ``f``.

Under these conditions, the rejection sampling algorithm given below will
generate independent samples from ``f``:

* Sample ``X \sim g``
* Accept ``X`` with probability ``\frac{f(X)}{Mg(X)}``

# Arguments
* `f::Function`: density function for the target distribution.
* `g::Function`: density function for the proposal distribution.
* `proposal_sampler::Function`: a function that generates samples from the
    proposal distribution.
* `M::Real`: a scaling constant that bounds the ratio of the target and
    proposal density.

# Notes
* No checks are made to ensure that the scale constant is valid or that the
  proposal sampler and density match. If these conditions are not met, the
  resulting samples will not be distributed according to the target density.
* The expected acceptance rate for the sampler is the reciprocal of `M`.

# Examples
```julia
# Generating 10 standard normal samples using a Cauchy proposal
f(x) = exp(-x^2 / 2) / sqrt(2π)
g(x) = 1 / (π * (1 + x^2))
proposal_sampler = InverseTransformSampler(u -> tan(π * (u - 0.5)))
M = sqrt(2π / ℯ)
s = RejectionSampler(f, g, proposal_sampler, M)
samples = sample(s, 10)
```
"""
struct RejectionSampler <: Sampler
    f::Function
    g::Function
    proposal_sampler::Sampler
    M::Real
end

@doc raw"""
    sample(s::RejectionSampler, N)

Generate samples using a rejection sampler.

# Arguments
* `s::RejectionSampler`: a rejection sampler.
* `N::Integer`: the number of samples to generate.

# Returns
* `samples::Array{Float64}`: a 2-D array in which each column is a sample.
"""
function sample(s::RejectionSampler, N::Integer)
    # TODO: make this work for arbritary dimensions
    samples = Array{Float64}(undef, 1, N)
    i = 1
    while i <= N
        proposal = sample(s.proposal_sampler, 1)[1]

        threshold = s.f(proposal) / (s.M * s.g(proposal))
        if rand() < threshold
            samples[:, i] .= proposal
            i += 1
        end
    end

    return samples
end

@doc raw"""
    InverseTransformSampler(F_inv)

An inverse transform sampler.

Inverse transform sampling is based on the result that when
``U \sim \text{Unif}(0, 1)``, ``X = F^{-1}(U)`` will be distributed
according to the CDF ``F``. For non-decreasing CDFs we can replace ``F^{-1}``
with ``F``'s generalised inverse ``F^{-}(u) = \inf{x : F(x) > u}`` and use
the same method.

# Arguments
* `F_inv::Function`: the inverse CDF of the target distrubition.

# Notes
* Because of the symmetry of a uniform random sample, one can replace ``1-u``
  in the inverse CDF with ``u``.

# Examples
```julia
# Generating 10 standard exponential random variables
F_inv(u) = -log(u)  # using symmetry of U (see notes)
s = InverseTransformSampler(F_inv)
samples = sample(s, 10)
```
"""
struct InverseTransformSampler <: Sampler
    F_inv::Function
end

@doc raw"""
    sample(s::InverseTransformSampler, N)

Generate samples using an inverse transform sampler.

# Arguments
* `s::InverseTransformSampler`: an inverse transform sampler.
* `N::Integer`: the number of samples to generate.

# Returns
* `samples::Array{Float64}`: a 2-D array in which each column is a sample.
"""
function sample(s::InverseTransformSampler, N::Integer)
    return s.F_inv.(rand(1, N))
end

@doc raw"""
    BoxMullerTransformSampler()

A Box-Muller transform sampler.

The Box-Muller transform is a technique for generating independent, standard,
normally distributed samples. It involves sampling from a polar coordinate
system in which the angle of the sample is uniformly distrubition and the
radius is distributed as ``\text{exp}(\tfrac{1}{2})`` (equivalent to
``\chi^2_1``). It can then be shown that the cartesian coordinates of these
samples have independent standard normal distributions.

The full algorithm is given here:

* Sample ``U_1, U_2 \stackrel{\text{i.i.d.}}{\sim} \text{Unif}(0, 1)``
* Compute ``Z_1 = \sqrt{-2 \log U_1}\cos(2\pi U_2)``, ``Z_2 = \sqrt{-2 \log U_1}\sin(2\pi U_2)``

# Arguments
_None_

# Examples
```julia
# Generating 10 standard normal samples
s = BoxMullerTransformSampler()
samples = sample(s, 10)
```
"""
struct BoxMullerTransformSampler <: Sampler
end

@doc raw"""
    sample(s::BoxMullerTransformSampler, N)

Generate samples using a Box-Muller transform sampler.

# Arguments
* `s::BoxMullerTransformSampler`: a Box-Muller transform sampler.
* `N::Integer`: the number of samples to generate.

# Returns
* `samples::Array{Float64}`: a 2-D array in which each column is a sample.
"""
function sample(s::BoxMullerTransformSampler, N::Integer)
    M = N + N % 2
    U1 = rand(1, M)
    U2 = rand(1, M)
    R = sqrt.(-2 * log.(U1))
    θ = 2π * U2
    Z1 = R .* cos.(θ)
    Z2 = R .* sin.(θ)
    return hcat(Z1, Z2)[:, 1:N]
end
