var documenterSearchIndex = {"docs":
[{"location":"index.html#MonteCarloZoo.jl","page":"Index","title":"MonteCarloZoo.jl","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"Documentation for MonteCarloZoo.jl.","category":"page"},{"location":"index.html","page":"Index","title":"Index","text":"MonteCarloZoo","category":"page"},{"location":"index.html#MonteCarloZoo","page":"Index","title":"MonteCarloZoo","text":"MonteCarloZoo\n\nA broad collection of Monte Carlo algorithms implemented in Julia.\n\n\n\n\n\n","category":"module"},{"location":"index.html#Module-Index","page":"Index","title":"Module Index","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"Modules = [MonteCarloZoo]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"index.html#Detailed-API","page":"Index","title":"Detailed API","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"Modules = [MonteCarloZoo]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"index.html#MonteCarloZoo.BoxMullerTransformSampler","page":"Index","title":"MonteCarloZoo.BoxMullerTransformSampler","text":"BoxMullerTransformSampler()\n\nA Box-Muller transform sampler.\n\nThe Box-Muller transform is a technique for generating independent, standard, normally distributed samples. It involves sampling from a polar coordinate system in which the angle of the sample is uniformly distrubition and the radius is distributed as textexp(tfrac12) (equivalent to chi^2_1). It can then be shown that the cartesian coordinates of these samples have independent standard normal distributions.\n\nThe full algorithm is given here:\n\nSample U_1 U_2 stackreltextiidsim textUnif(0 1)\nCompute Z_1 = sqrt-2 log U_1cos(2pi U_2), Z_2 = sqrt-2 log U_1sin(2pi U_2)\n\nArguments\n\nNone\n\nExamples\n\n# Generating 10 standard normal samples\ns = BoxMullerTransformSampler()\nsamples = sample(s, 10)\n\n\n\n\n\n","category":"type"},{"location":"index.html#MonteCarloZoo.InverseTransformSampler","page":"Index","title":"MonteCarloZoo.InverseTransformSampler","text":"InverseTransformSampler(F_inv)\n\nAn inverse transform sampler.\n\nInverse transform sampling is based on the result that when U sim textUnif(0 1), X = F^-1(U) will be distributed according to the CDF F. For non-decreasing CDFs we can replace F^-1 with F's generalised inverse F^-(u) = infx  F(x)  u and use the same method.\n\nArguments\n\nF_inv::Function: the inverse CDF of the target distrubition.\n\nNotes\n\nBecause of the symmetry of a uniform random sample, one can replace 1-u in the inverse CDF with u.\n\nExamples\n\n# Generating 10 standard exponential random variables\nF_inv(u) = -log(u)  # using symmetry of U (see notes)\ns = InverseTransformSampler(F_inv)\nsamples = sample(s, 10)\n\n\n\n\n\n","category":"type"},{"location":"index.html#MonteCarloZoo.LinearCongruentialGenerator","page":"Index","title":"MonteCarloZoo.LinearCongruentialGenerator","text":"LinearCongruentialGenerator(a, c, m, seed)\n\nA linear congruential generator.\n\nA linear congruential generator (LCG) is a primitive psueudo-random number generator (PRNG), producing samples that approximate a textUniform(0 1) distribution. Starting from a seed X_0, a sequence is generated according to the recurence relation,\n\nX_n+1 = (aX_n + c) mod m\n\nwhich are then turned into samples Z_n = fracX_nm in 0 1).\n\nAlthough the generator is favoured for esoteric purposes due to its simplicity and speed, in practice it is highly flawed and can result in samples that have minimal random structure when viewed through an appropriate projection (as in Entacher). The generator is also highly sensitive to the choice of a, c, and m.\n\nArguments\n\na::Integer: the \"multiplier\"; must be between 0 and m, exclusive.\nc::Integer: the \"increment\"; must be between 0 and m-1, inclusive.\nm::Integer: the \"modulus\"; must be non-negative   proposal distribution.\nseed::Integer: the seed value; must be between 0 and m-1 inclusive\n\nExamples\n\n# Generating 10 uniform random variables\n# Parameters taken from \"Numerical Recipes\"\na = 1664525\nc = 1013904223\nm = 2^32\nseed = 1729\ns = LinearCongruentialGenerator(a, c, m, seed)\nsamples = sample(s, 10)\n\n# Batch sampling\ns = LinearCongruentialGenerator(a, c, m, seed)\nsamples1 = sample(s, 10)\nsamples2 = sample(s, 10)\n# samples1 ≠ samples2 in general\n\n\n\n\n\n","category":"type"},{"location":"index.html#MonteCarloZoo.NormalNormalTransformer","page":"Index","title":"MonteCarloZoo.NormalNormalTransformer","text":"NormalNormalTransformer(normal_sampler, new_params, old_params=(0.0, 1.0))\n\nGenerate normal random variables from another normal sampler.\n\nTransform samples from a normal distribution sampler into samples from a normal distribution with different parameters. This uses the result that if Z is a standard normal random variable (Z sim textNormal(0 1)) then X =  sigma Z + mu has distribution textNormal(mu sigma^2).\n\nArguments\n\nnormal_sampler::sampler: a sampler producing normally distributed samples.\nnew_params::Tuple{Real, Real}: a tuple (mu, sigma2) giving the mean and   variance of the target distribution\nold_params::Tuple{Real, Real}=(0.0, 1.0): a tuple (mu, sigma2) giving the   mean and variance of the normal sampler. Defaults to a standard normal   distribution.\n\nExamples\n\n# Transforming ten standard normal random variables into N(1, 2)\ns = BoxMullerTransformSampler()\nt = NormalNormalTransformer(s, (1, 2))\ntransformed_samples = sample(t, 10)\n\n\n\n\n\n","category":"type"},{"location":"index.html#MonteCarloZoo.RejectionSampler","page":"Index","title":"MonteCarloZoo.RejectionSampler","text":"RejectionSampler(f, g, proposal_sampler, M)\n\nA rejection sampler.\n\nRejection sampling involves sampling from a target distribution f using a proposal distribution g which we are able to directly sample from. In order for the resulting samples to be distributed according to f we require that fracf(x)g(x) is bounded from above by some constant M (M) for all x in the support of f.\n\nUnder these conditions, the rejection sampling algorithm given below will generate independent samples from f:\n\nSample X sim g\nAccept X with probability fracf(X)Mg(X)\n\nArguments\n\nf::Function: density function for the target distribution.\ng::Function: density function for the proposal distribution.\nproposal_sampler::Function: a function that generates samples from the   proposal distribution.\nM::Real: a scaling constant that bounds the ratio of the target and   proposal density.\n\nNotes\n\nNo checks are made to ensure that the scale constant is valid or that the proposal sampler and density match. If these conditions are not met, the resulting samples will not be distributed according to the target density.\nThe expected acceptance rate for the sampler is the reciprocal of M.\n\nExamples\n\n# Generating 10 standard normal samples using a Cauchy proposal\nf(x) = exp(-x^2 / 2) / sqrt(2π)\ng(x) = 1 / (π * (1 + x^2))\nproposal_sampler = InverseTransformSampler(u -> tan(π * (u - 0.5)))\nM = sqrt(2π / ℯ)\ns = RejectionSampler(f, g, proposal_sampler, M)\nsamples = sample(s, 10)\n\n\n\n\n\n","category":"type"},{"location":"index.html#MonteCarloZoo.sample-Tuple{BoxMullerTransformSampler, Integer}","page":"Index","title":"MonteCarloZoo.sample","text":"sample(s::BoxMullerTransformSampler, N)\n\nGenerate samples using a Box-Muller transform sampler.\n\nArguments\n\ns::BoxMullerTransformSampler: a Box-Muller transform sampler.\nN::Integer: the number of samples to generate.\n\nReturns\n\nsamples::Array{Float64}: a 2-D array in which each column is a sample.\n\n\n\n\n\n","category":"method"},{"location":"index.html#MonteCarloZoo.sample-Tuple{InverseTransformSampler, Integer}","page":"Index","title":"MonteCarloZoo.sample","text":"sample(s::InverseTransformSampler, N)\n\nGenerate samples using an inverse transform sampler.\n\nArguments\n\ns::InverseTransformSampler: an inverse transform sampler.\nN::Integer: the number of samples to generate.\n\nReturns\n\nsamples::Array{Float64}: a 2-D array in which each column is a sample.\n\n\n\n\n\n","category":"method"},{"location":"index.html#MonteCarloZoo.sample-Tuple{LinearCongruentialGenerator, Integer}","page":"Index","title":"MonteCarloZoo.sample","text":"sample(s::LinearCongruentialGenerator, N)\n\nGenerate samples using a linear congruential generator.\n\nArguments\n\ns::LinearCongruentialGenerator: a linear congruential generator.\nN::Integer: the number of samples to generate.\n\nReturns\n\nsamples::Array{Float64}: a 2-D array in which each column is a sample.\n\n\n\n\n\n","category":"method"},{"location":"index.html#MonteCarloZoo.sample-Tuple{NormalNormalTransformer, Integer}","page":"Index","title":"MonteCarloZoo.sample","text":"sample(s::NormalNormalTransformer, N)\n\nTransform normal samples into normal samples with new parameters.\n\nArguments\n\ns::NormalNormalTransformer: a normal-normal transformer.\nN::Integer: the number of samples to generate.\n\nReturns\n\nsamples::Array{Float64}: a 2-D array in which each column is a sample.\n\n\n\n\n\n","category":"method"},{"location":"index.html#MonteCarloZoo.sample-Tuple{RejectionSampler, Integer}","page":"Index","title":"MonteCarloZoo.sample","text":"sample(s::RejectionSampler, N)\n\nGenerate samples using a rejection sampler.\n\nArguments\n\ns::RejectionSampler: a rejection sampler.\nN::Integer: the number of samples to generate.\n\nReturns\n\nsamples::Array{Float64}: a 2-D array in which each column is a sample.\n\n\n\n\n\n","category":"method"}]
}
