module RejectionSampler

export rejection_sampler

function rejection_sampler(target_density, proposal_density, proposal_sampler,
                           scale, iterations, dimension)
    samples = Array{Float64}(undef, dimension, iterations)
    
    i = 1
    while i <= iterations
        proposal = proposal_sampler()
        threshold = target_density(proposal) / (scale * proposal_density(proposal))
        if rand() < threshold
            samples[:, i] = proposal
            i += 1
        end
    end

    return samples
end

end  # module
