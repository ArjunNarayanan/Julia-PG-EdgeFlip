module Policy

using Flux
include("PPO_edge_model.jl")


struct PolicyNL
    emodels::Any
    bvals::Any
    lmodel::Any
    num_levels::Any
    num_hidden_channels::Any
    function PolicyNL(num_levels, num_hidden_channels)
        emodels = [EdgeModel(4, num_hidden_channels)]
        for i = 2:num_levels
            push!(emodels, EdgeModel(num_hidden_channels, num_hidden_channels))
        end

        bvals = Flux.glorot_uniform(num_hidden_channels)
        lmodel = Dense(num_hidden_channels, 1)
        new(emodels, bvals, lmodel, num_levels, num_hidden_channels)
    end
end

Flux.@functor PolicyNL

function Base.show(io::IO, policy::PolicyNL)
    s = "PolicyNL\n\t $(policy.num_levels) levels\n\t $(policy.num_hidden_channels) channels"
    println(io, s)
end

function eval_single(p::PolicyNL, ep, epairs)
    x = eval_single(p.emodels[1], ep, epairs, p.bvals)
    x = relu.(x)

    for i in 2:length(p.emodels)
        y = eval_single(p.emodels[i], x, epairs, p.bvals)
        y = x + y
        x = relu.(y)
    end

    logits = vec(p.lmodel(x))
    return logits
end

function eval_batch(p::PolicyNL, ep, epairs)
    nf, na, nb = size(ep)
    ep = reshape(ep, nf, :)
    epairs = vec(epairs)

    logits = eval_single(p, ep, epairs)
    logits = reshape(logits, na, nb)
    return logits
end

end