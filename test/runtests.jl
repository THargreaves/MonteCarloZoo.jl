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
