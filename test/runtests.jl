using Test, MonteCarloZoo, Distributions, Random, Plots, SpecialFunctions

function ks_test(samples, target_cdf)
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

## Rejection Sampling

target_density(x) = exp(-(x^2 / 2)) / sqrt(2π)  # standard normal
proposal_density(x) = 1 / (π * (1 + x^2))  # Cauchy
proposal_sampler() = tan(π * (rand() - 0.5))
scale = sqrt(2π / ℯ)
iterations = 1000
dimension = 1

Random.seed!(1729)
samples = rejection_sampler(
    target_density, proposal_density, proposal_sampler,
    scale, iterations, dimension
)

@test size(samples) == (dimension, iterations)
p = ks_test(samples, x -> 1 / 2 * (1 + erf(x / sqrt(2))))
@test p > 0.05

## Inverse Transform Sampling

F_inv(u) = -log(u)  # using symmetry of U (see notes)
N = 1000

Random.seed!(1729)
samples = inverse_transform_sampler(F_inv, N)

p = histogram(samples[1, :])
savefig("out.png")

@test size(samples) == (1, N)
p = ks_test(samples, x -> 1 - exp(-x))
@test p > 0.05
