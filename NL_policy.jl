using Flux
include("edge_policy_gradient.jl")
include("edge_model.jl")

PG = EdgePolicyGradient

struct PolicyNL
    emodels
    lmodel::Any
    function PolicyNL(num_levels, num_hidden_channels)
        emodels = [EdgeModel(4,num_hidden_channels)]
        for i in 2:num_levels
            push!(emodels, EdgeModel(num_hidden_channels, num_hidden_channels))
        end
        
        lmodel = Dense(16, 1)
        new(emodels, lmodel)
    end
end

Flux.@functor PolicyNL

function PG.eval_single(p::PolicyNL, ep, econn, epairs)
    x = eval_single(p.emodels[1], ep, econn, epairs)
    x = relu.(x)

    for i in 2:length(p.emodels)
        y = eval_single(p.emodels[i], x, econn, epairs)
        y = x + y
        x = relu.(y)
    end
    
    logits = p.lmodel(x)
    return logits
end

function PG.eval_batch(p::PolicyNL, ep, econn, epairs)
    x = eval_batch(p.emodels[1], ep, econn, epairs)
    x = relu.(x)

    for i in 2:length(p.emodels)
        y = eval_batch(p.emodels[i], x, econn, epairs)
        y = x + y
        x = relu.(y)
    end

    logits = p.lmodel(x)
    return logits
end