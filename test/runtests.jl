using Test, MonteCarloZoo, Random, Plots, SpecialFunctions
import Distributions: Kolmogorov, cdf

function ks_test(samples, target_cdf)
    sorted_samples = sort(samples, dims=2)
    D = 0
    for (i, (s, t)) in enumerate(zip(sorted_samples[:, 1:end - 1],
                                     sorted_samples[:, 1:end]))
        D = max(D,
                abs(target_cdf(s) - i / length(samples)),
                abs(target_cdf(t) - i / length(samples)))
    end
    K = D * sqrt(length(samples))
    p = 1 - cdf(Kolmogorov(), K)
    return p
end

## Rejection Sampling

f(x) = exp(-x^2 / 2) / sqrt(2π)
g(x) = 1 / (π * (1 + x^2))
proposal_sampler = InverseTransformSampler(u -> tan(π * (u - 0.5)))
M = sqrt(2π / ℯ)
N = 1000

s = RejectionSampler(f, g, proposal_sampler, M)
Random.seed!(1729)
samples = sample(s, N)

@test size(samples) == (1, N)
p = ks_test(samples, x -> 1 / 2 * (1 + erf(x / sqrt(2))))
@test p > 0.05

## Inverse Transform Sampling

F_inv(u) = -log(u)  # using symmetry of U (see notes)
N = 1000

s = InverseTransformSampler(F_inv)
Random.seed!(1729)
samples = sample(s, N)

@test size(samples) == (1, N)
p = ks_test(samples, x -> 1 - exp(-x))
@test p > 0.05

## Box-Muller Transform Sampling

function test_box_muller()
    for N in (1000, 1001)
        Random.seed!(1729)
        s = BoxMullerTransformSampler()
        samples = sample(s, N)

        @test size(samples) == (1, N)
        p = ks_test(samples, x -> 1 / 2 * (1 + erf(x / sqrt(2))))
        @test p > 0.05
    end
end

test_box_muller()

## Linear Congruential Generator

a = 1664525
c = 1013904223
m = 2^32
seed = 1729
N = 1000

s = LinearCongruentialGenerator(a, c, m, seed)
samples = sample(s, N)

@test size(samples) == (1, N)
p = ks_test(samples, x -> x * (0 ≤ x ≤ 1))
@test p > 0.05

## Normal-Normal Transformer

μ₁ = 1.0
σ² = 2.0
normal_sampler = BoxMullerTransformSampler()
N = 1000

t1 = NormalNormalTransformer(normal_sampler, (μ₁, σ²))
Random.seed!(1729)
samples = sample(t1, N)

@test size(samples) == (1, N)
p = ks_test(samples, x -> 1 / 2 * (1 + erf((x - μ₁) / sqrt(2 * σ²))))
@test p > 0.05

μ₂ = 2.0

t2 = NormalNormalTransformer(t1, (μ₂, σ²), (μ₁, σ²))
Random.seed!(1729)
samples = sample(t2, N)

@test size(samples) == (1, N)
p = ks_test(samples, x -> 1 / 2 * (1 + erf((x - μ₂) / sqrt(2 * σ²))))
@test p > 0.05
