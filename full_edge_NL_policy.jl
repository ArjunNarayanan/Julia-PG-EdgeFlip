using Flux
include("full_edge_model.jl")
include("full_edge_policy_gradient.jl")

PG = FullEdgePolicyGradient

struct FullEdgePolicyNL
    emodels::Any
    lmodel::Any
    num_levels::Any
    num_hidden_channels::Any
    function FullEdgePolicyNL(num_levels, num_hidden_channels)
        emodels = [FullEdgeModel(4, num_hidden_channels)]
        for i = 2:num_levels
            push!(emodels, FullEdgeModel(num_hidden_channels, num_hidden_channels))
        end

        lmodel = Dense(num_hidden_channels, 1)
        new(emodels, lmodel, num_levels, num_hidden_channels)
    end
end

function PG.eval_single(p::FullEdgePolicyNL, ep, econn)
    x = eval_single(p.emodels[1], ep, econn)
    x = relu.(x)

    for i in 2:length(p.emodels)
        y = eval_single(p.emodels[i], x, econn)
        y = x + y
        x = relu.(y)
    end

    logits = p.lmodel(x)
    return logits
end

function PG.eval_batch(p::FullEdgePolicyNL, ep, econn)
    x = eval_batch(p.emodels[1], ep, econn)
    x = relu.(x)

    for i in 2:length(p.emodels)
        y = eval_batch(p.emodels[i], x, econn)
        y = x + y
        x = relu.(y)
    end

    logits = p.lmodel(x)
    return logits
end

Flux.@functor FullEdgePolicyNL

function Base.show(io::IO, policy::FullEdgePolicyNL)
    s = "PolicyNL\n\t $(policy.num_levels) levels\n\t $(policy.num_hidden_channels) channels"
    println(io, s)
end