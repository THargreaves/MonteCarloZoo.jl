"""
    MonteCarloZoo

A broad collection of Monte Carlo algorithms implemented in Julia.
"""
module MonteCarloZoo

include("Diagnostics.jl")
include("Filtering.jl")
include("Generators.jl")
include("MarkovChain.jl")
include("Simple.jl")
include("Transformation.jl")
include("Utils.jl")

end  # module
