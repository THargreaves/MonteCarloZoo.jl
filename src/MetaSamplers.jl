using Random
Random.seed!(1)

abstract type MetaSampler end
# abstract type AbstractSampler end

struct Sampler end

struct GibbsSampler <: MetaSampler
    initial_value
    samplers::Tuple{Sampler}
    x
    GibbsSampler(initial_value, samplers) = GibbsSampler(initial_value, samplers, initial_value)
end

struct CompositionSampler <: MetaSampler
    samplers::Tuple{Sampler}
end


S1 = Sampler(1)
S2 = Sampler(2)
GibbsSampler((0, 1, 2), (S1, S2))

function sample(sampler::GibbsSampler)
    i = 1
    for s in sampler.samplers
        d = s.dim
        x[i:i+d] = sample(s, vcat(x[1:i-1], x[i+d+1:end]))
        i += d
    end
    return x
end

function sample(sampler::CompositionSampler)
    i = 1
    for s in sampler.samplers
        d = s.dim
        x[i:i+d] = sample(s, x[1:i-1])
        i += d
    end
    return x
end

# X2, X1 ~ N(0, I2)
# X3 | X2, X1 ~ N(X2 - X1, 1)

struct Parameter 
    name::String
end

s = BoxMullerTransformSampler()  # samples N(0, 1)
t1 = NormalNormalTransformer(s, (0, 2))
t2 = NormalNormalTransformer(s, (Parameter("σ") + 1, 3))  # X1 not defined yet

# Replace X1 with Parameter(1) or Parameter("X1")

# When sampler gets to a parameter, it sees if it (the sampler) has a meta
# sampler, if not then throw error, otherwise checks if param exists, if not
# then throw error, otherwise use its value.

(x1 = 2, x2 = 3)

sampling (X1, X2)  ~ N2((0, 1), (2, c,
                                 c, 3))

CompositionSampler((t1, t2))
