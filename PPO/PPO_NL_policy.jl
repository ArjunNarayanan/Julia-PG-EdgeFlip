module Policy

using Flux
include("../edge_model.jl")


struct PolicyNL
    emodels
    lmodel::Any
    num_levels
    num_hidden_channels
    function PolicyNL(num_levels, num_hidden_channels)
        emodels = [EdgeModel(4, num_hidden_channels)]
        for i in 2:num_levels
            push!(emodels, EdgeModel(num_hidden_channels, num_hidden_channels))
        end

        lmodel = Dense(num_hidden_channels, 1)
        new(emodels, lmodel, num_levels, num_hidden_channels)
    end
end

Flux.@functor PolicyNL

function Base.show(io::IO, policy::PolicyNL)
    s = "PolicyNL\n\t $(policy.num_levels) levels\n\t $(policy.num_hidden_channels) channels"
    println(io, s)
end

function eval_single(p::PolicyNL, ep, econn, epairs)
    x = eval_single(p.emodels[1], ep, econn, epairs)
    x = relu.(x)

    for i in 2:length(p.emodels)
        y = eval_single(p.emodels[i], x, econn, epairs)
        y = x + y
        x = relu.(y)
    end

    logits = vec(p.lmodel(x))

    return logits
end

function eval_batch(p::PolicyNL, ep, econn, epairs)
    x = eval_batch(p.emodels[1], ep, econn, epairs)
    x = relu.(x)

    for i in 2:length(p.emodels)
        y = eval_batch(p.emodels[i], x, econn, epairs)
        y = x + y
        x = relu.(y)
    end

    nf, na, nb = size(x)
    logits = reshape(p.lmodel(x), na, nb)
    return logits
end


end