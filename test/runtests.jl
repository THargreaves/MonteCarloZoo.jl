using Test, MonteCarloZoo, Distributions, Random, Plots, SpecialFunctions

target_density(x) = exp(-(x^2 / 2)) / sqrt(2π)  # standard normal
proposal_density(x) = 1 / (π * (1 + x^2))  # Cauchy
proposal_sampler() = tan(π * (rand() - 0.5))
scale = sqrt(2π / ℯ)
iterations = 500
dimension = 1

Random.seed!(1729)
samples = rejection_sampler(
    target_density, proposal_density, proposal_sampler,
    scale, iterations, dimension
)

@test size(samples) == (dimension, iterations)

# Kolmogorov-Smirnov Test
target_cdf(x) = 1 / 2 * (1 + erf(x / sqrt(2)))

function ks_test()
    sorted_samples = sort(samples, dims=2)
    D = 0
    for (i, (s, t)) in enumerate(zip(sorted_samples[:, 1:end - 1],
                                     sorted_samples[:, 1:end]))
        D = max(D,
                abs(target_cdf(s) - i / iterations),
                abs(target_cdf(t) - i / iterations))
    end
    K = D * sqrt(iterations)
    p = 1 - cdf(Kolmogorov(), K)
    return p
end

p = ks_test()
@test p > 0.05
