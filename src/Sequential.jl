export rejection_sampler
export inverse_transform_sampler
export box_muller_transform_sampler


@doc raw"""
    rejection_sampler(target_density, proposal_density, proposal_sampler,
                      scale, iterations, dimension)

Perform rejection sampling using a supplied target and proposal distribution.

Rejection sampling involves sampling from a target distribution ``f`` using a
proposal distribution ``g`` which we are able to directly sample from. In order
for the resulting samples to be distributed according to ``f`` we require that
``\frac{f(x)}{g(x)}`` is bounded from above by some constant ``M`` (`scale`) for all ``x`` in
the support of ``f``.

Under these conditions, the rejection sampling algorithm given below will
generate independent samples from ``f``:

* Sample ``X \sim g``
* Accept ``X`` with probability ``\frac{f(X)}{Mg(X)}``

# Arguments
* `target_density::Function`: density function for the target distribution.
* `proposal_density::Function`: density function for the proposal distribution.
* `proposal_sampler::Function`: a function that generates samples from the
    proposal distribution.
* `scale::Real`: a scaling constant that bounds the ratio of the target and
    proposal density.
* `iterations::Integer`: the number of samples to generate.
* `dimension::Integer`: the expected dimension of each sample.

# Notes
* No checks are made to ensure that the scale constant is valid or that the 
  proposal sampler and density match. If these conditions are not met, the
  resulting samples will not be distributed according to the target density.
* The expected acceptance rate for the sampler is the reciprocal of `scale`.

# Examples
```julia
# Sampling from a standard normal distribution using Cauchy proposals
target_density(x) = exp(-x^2 / 2) / sqrt(2π)
proposal_density(x) = (1 + x^2) / π 
proposal_sampler() = tan(π * (rand() - 0.5))
scale = sqrt(2π / ℯ)
iterations = 100
dimension = 1

samples = rejection_sampler(
    target_density, proposal_density, proposal_sampler,
    scale, iterations, dimension
)
```
"""
function rejection_sampler(target_density, proposal_density, proposal_sampler,
                           scale, iterations, dimension)
    samples = Array{Float64}(undef, dimension, iterations)
    
    i = 1
    while i <= iterations
        proposal = proposal_sampler()
        threshold = target_density(proposal) / (scale * proposal_density(proposal))
        if rand() < threshold
            samples[:, i] .= proposal
            i += 1
        end
    end

    return samples
end

@doc raw"""
    inverse_transform_sampler(F_inv, N)

Perform inverse transform sampling using a supplied inverse CDF.

Inverse transform sampling is based on the result that when
``U \sim \text{Unif}(0, 1)``, ``X = F^{-1}(U)`` will be distributed
according to the CDF ``F``. For non-decreasing CDFs we can replace ``F^{-1}``
with ``F``'s generalised inverse ``F^{-}(u) = \inf{x : F(x) > u}`` and use
the same method.

# Arguments
* `F_inv::Function`: the inverse CDF of the target distrubition.
* `N::Integer`: the number of samples to generate.

# Notes
* Because of the symmetry of a uniform random sample, one can replace ``1-u``
  in the inverse CDF with ``u``.

# Examples
```julia
# Sampling from an exponential distribution
F_inv(u) = -log(u)  # using symmetry of U (see notes)
N = 100

samples = inverse_transform_sampler(F_inv, N)
```
"""
function inverse_transform_sampler(F_inv, N)
    return F_inv.(rand(1, N))
end

@doc raw"""
    box_muller_transform_sampler(N)

Perform Box-Muller transform sampling to generate standard Normal samples.

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
* `N::Integer`: the number of samples to generate.

# Examples
```julia
# 10 standard normal random variables
samples = box_muller_transform_sampler(10)
```
"""
function box_muller_transform_sampler(N)
    M = N + N % 2
    U1 = rand(1, M)
    U2 = rand(1, M)
    R = sqrt.(-2 * log.(U1))
    θ = 2π * U2
    Z1 = R .* cos.(θ)
    Z2 = R .* sin.(θ)
    return hcat(Z1, Z2)[:, 1:N]
end
