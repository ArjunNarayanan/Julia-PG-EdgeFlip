using Flux
include("edge_policy_gradient.jl")
include("unaveraged_edge_model.jl")

PG = EdgePolicyGradient

struct PolicyNL
    emodels
    lmodel::Any
    num_levels
    num_hidden_channels
    function PolicyNL(num_levels, num_hidden_channels)
        emodels = [UnaveragedEdgeModel(4, num_hidden_channels)]
        for i in 2:num_levels
            push!(emodels, UnaveragedEdgeModel(num_hidden_channels, num_hidden_channels))
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
