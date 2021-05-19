export NormalNormalTransformer

@doc raw"""
    NormalNormalTransformer(normal_sampler, new_params, old_params=(0.0, 1.0))

Generate normal random variables from another normal sampler.

Transform samples from a normal distribution sampler into samples from a normal
distribution with different parameters. This uses the result that if ``Z`` is a
standard normal random variable (``Z \sim \text{Normal}(0, 1)``) then
``X =  \sigma Z + \mu`` has distribution ``\text{Normal}(\mu, \sigma^2)``.

# Arguments
* `normal_sampler::sampler`: a sampler producing normally distributed samples.
* `new_params::Tuple{Real, Real}`: a tuple (mu, sigma2) giving the mean and
    variance of the target distribution
* `old_params::Tuple{Real, Real}=(0.0, 1.0)`: a tuple (mu, sigma2) giving the
    mean and variance of the normal sampler. Defaults to a standard normal
    distribution.

# Examples
```julia
# Transforming ten standard normal random variables into N(1, 2)
s = BoxMullerTransformSampler()
t = NormalNormalTransformer(s, (1, 2))
transformed_samples = sample(t, 10)
```
"""
struct NormalNormalTransformer <: Sampler
    normal_sampler::Sampler
    new_params::NamedTuple{(:mu, :sigma2), Tuple{Real, Real}}
    old_params::NamedTuple{(:mu, :sigma2), Tuple{Real, Real}}
    function NormalNormalTransformer(
        normal_sampler::Sampler,
        new_params::Tuple{Real, Real},
        old_params::Tuple{Real, Real}=(0.0, 1.0)
    )
        # Name parameter tuples
        new_params = (mu = new_params[1], sigma2 = new_params[2])
        old_params = (mu = old_params[1], sigma2 = old_params[2])
        # Validation
        if !(old_params[:sigma2] > 0 && new_params[:sigma2] > 0)
            error("variance must be positive")
        end
        new(normal_sampler, new_params, old_params)
    end
end

@doc raw"""
    sample(s::NormalNormalTransformer, N)

Transform normal samples into normal samples with new parameters.

# Arguments
* `s::NormalNormalTransformer`: a normal-normal transformer.
* `N::Integer`: the number of samples to generate.

# Returns
* `samples::Array{Float64}`: a 2-D array in which each column is a sample.
"""
function sample(s::NormalNormalTransformer, N::Integer)
    old_samples = sample(s.normal_sampler, N)
    std_samples = (old_samples .- s.old_params[:mu]) ./
        sqrt(s.old_params[:sigma2])
    new_samples = sqrt(s.new_params[:sigma2]) .* std_samples .+
        s.new_params[:mu]

    return new_samples
end
