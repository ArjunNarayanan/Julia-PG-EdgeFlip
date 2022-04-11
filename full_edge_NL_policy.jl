using Flux
include("full_edge_model.jl")

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

function (p::FullEdgePolicyNL)(ep, econn)
    x = p.emodels[1](ep, econn)
    x = relu.(x)

    for i = 2:length(p.emodels)
        y = p.emodels[i](x, econn)
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