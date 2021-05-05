using Test, MonteCarloZoo, Distributions, Random, Plots, SpecialFunctions

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

f(x) = exp(-(x^2 / 2)) / sqrt(2π)  # standard normal
g(x) = 1 / (π * (1 + x^2))  # Cauchy
proposal_sampler() = tan(π * (rand() - 0.5))
M = sqrt(2π / ℯ)
N = 1000
dimension = 1

Random.seed!(1729)
samples = rejection_sampler(f, g, proposal_sampler, M, N, dimension)

@test size(samples) == (dimension, N)
p = ks_test(samples, x -> 1 / 2 * (1 + erf(x / sqrt(2))))
@test p > 0.05

## Inverse Transform Sampling

F_inv(u) = -log(u)  # using symmetry of U (see notes)
N = 1000

Random.seed!(1729)
samples = inverse_transform_sampler(F_inv, N)

@test size(samples) == (1, N)
p = ks_test(samples, x -> 1 - exp(-x))
@test p > 0.05

## Box-Muller Transform Sampling

function test_box_muller()
    for N in (1000, 1001)
        Random.seed!(1729)
        samples = box_muller_transform_sampler(N)

        @test size(samples) == (1, N)
        p = ks_test(samples, x -> 1 / 2 * (1 + erf(x / sqrt(2))))
        @test p > 0.05
    end
end

test_box_muller()
