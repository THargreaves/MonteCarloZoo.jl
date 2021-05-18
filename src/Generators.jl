export LinearCongruentialGenerator

@doc raw"""
    LinearCongruentialGenerator(a, c, m, seed)

A linear congruential generator.

A linear congruential generator (LCG) is a primitive psueudo-random number
generator (PRNG), producing samples that approximate a ``\text{Uniform}(0, 1)``
distribution. Starting from a seed ``X_0``, a sequence is generated according
to the recurence relation,

```math
X_{n+1} = (aX_n + c) \mod m
```

which are then turned into samples ``Z_n = \frac{X_n}{m} \in [0, 1)``.

Although the generator is favoured for esoteric purposes due to its simplicity
and speed, in practice it is highly flawed and can result in samples that have
minimal random structure when viewed through an appropriate projection (as in
[Entacher](https://dl.acm.org/doi/10.1145/272991.273009)). The generator is
also highly sensitive to the choice of ``a``, ``c``, and ``m``.

# Arguments
* `a::Integer`: the "multiplier"; must be between 0 and m, exclusive.
* `c::Integer`: the "increment"; must be between 0 and m-1, inclusive.
* `m::Integer`: the "modulus"; must be non-negative
    proposal distribution.
* `seed::Integer`: the seed value; must be between 0 and m-1 inclusive

# Examples
```julia
# Generating 10 uniform random variables
# Parameters taken from "Numerical Recipes"
a = 1664525
c = 1013904223
m = 2^32
seed = 1729
s = LinearCongruentialGenerator(a, c, m, seed)
samples = sample(s, 10)
```

```julia
# Batch sampling
s = LinearCongruentialGenerator(a, c, m, seed)
samples1 = sample(s, 10)
samples2 = sample(s, 10)
# samples1 ≠ samples2 in general
```
"""
struct LinearCongruentialGenerator <: Sampler
    a::Integer
    c::Integer
    m::Integer
    seed::Integer
    state::Vector{Int}
    function LinearCongruentialGenerator(
        a::Integer,
        c::Integer,
        m::Integer,
        seed::Integer
    )
        # Validation
        if !(0 < m)
            error("m must be positive")
        elseif !(0 < a < m)
            error("a must be between 1 and m, exclusive")
        elseif !(0 <= c < m)
            error("c must be between 0 and m-1, inclusive")
        elseif !(0 ≤ seed < m)
            error("seed must be between 0 and m-1, inclusive")
        end
        state = [seed]
        new(a, c, m, seed, state)
    end
end

@doc raw"""
    sample(s::LinearCongruentialGenerator, N)

Generate samples using a linear congruential generator.

# Arguments
* `s::LinearCongruentialGenerator`: a linear congruential generator.
* `N::Integer`: the number of samples to generate.

# Returns
* `samples::Array{Float64}`: a 2-D array in which each column is a sample.
"""
function sample(s::LinearCongruentialGenerator, N::Integer)
    samples = Array{Float64}(undef, 1, N)
    for i in 1:N
        s.state[1] = (s.a * s.state[1] + s.c) % s.m
        samples[i] = s.state[1] / s.m
    end

    return samples
end
