using SpecialFunctions

# Gibbs Sampler for 
μ = [0, 1, 2]
Σ = [1 0.5 0.1
    0.5 1 0.1
    0.1 0.1 1]
blocks = [(1, 2), (3)]
block_lengths = [2, 1]
# X ~ N(μ, Σ)
# with blocks [[X1, X2], [X3]] using inverse transform
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case_2

# X3 | X1, X2 ~ N((2 + 1 / 15(x1 + x2 - 1)), 99 / 100)
# X1, X2 | X3 ~ N((0.1 * x3 - 0.2, 0.1 * x3 + 0.8), ([0.99 0.49; 0.49 0.99]))

abstract type Sampler end

struct ParameterisedSampler <: Sampler
    sampler::Sampler
    parameters::NamedTuple
end


struct InverseTransformSampler <: Sampler
    F_inv::Function
end


struct NormalSampler <: Sampler
    parameters
    NormalSampler(parameters) = lineartransform(InverseTransformSampler(Φ⁻¹), parameters)
end

NormalSampler(mu, sigma) = NormalSampler((μ = mu, Σ = sigma))


abstract type MetaSampler end

struct GibbsSampler <: MetaSampler
    samplers::Tuple{N,Sampler} where {N} # N is the length of the tuple, "where N" just says it has no restrictions
    x0::AbstractVector
end


function lineartransform(sampler::Sampler, parameters::NamedTuple)::ParameterisedSampler
    parameter_names = keys(parameters)
    parameter_values = (val isa Function ? val : _ -> val for val in parameters)
    new_parameters = NamedTuple{parameter_names}(parameter_values)
    return ParameterisedSampler(sampler, new_parameters)
end


Φ⁻¹(p) = √2 * erfinv(2p - 1)
Phi_inv = Φ⁻¹


# Option 1
# X3Sampler = lineartransform(
#     InverseTransformSampler(Phi_inv),
#     (μ = (X1, X2) -> 2 + 1 / 15 * (X1 + X2 - 1), Σ = 99 / 100)
# )

# # Option 2
# X3Sampler = InverseTransformSampler(
#     (u, params) -> Phi_inv((u - params.mu) / params.sigma),
#     (mu = (X1, X2) -> 2 + 1 / 15(X1 + X2 - 1), sigma = 99 / 100)
# )

# # Option 3
# X3Sampler = InverseTransformSample(
#     (u, mu, sigma) -> Phi_inv((u - mu) / sigma),
#     (mu = (X1, X2) -> 2 + 1 / 15(X1 + X2 - 1), sigma = 99 / 100)
# )


# Option 4
X3Sampler = NormalSampler((
    mu = (X1, X2) -> 2 + 1 / 15(X1 + X2 - 1),
    sigma = 99 / 100
))

X1X2Sampler = NormalSampler((
    mu = X3 -> (0.1 * X3 - 0.2, 0.1 * X3 + 0.8),
    sigma = [0.99 0.49; 0.49 0.99]
))

gs = GibbsSampler((X1X2Sampler, X3Sampler), [0, 0, 0])