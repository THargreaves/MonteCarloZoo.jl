using Test, MonteCarloZoo

target_density(x) = exp(-x^2 / 2) / sqrt(2π)  # standard normal
proposal_density(x) = (1 + x^2) / π  # Cauchy
proposal_sampler() = tan(π * (rand() - 0.5))
scale = sqrt(2π / ℯ)
iterations = 100
dimension = 1

samples = rejection_sampler(
    target_density, proposal_density, proposal_sampler,
    scale, iterations, dimension
)

@test size(samples) == (dimension, iterations)
