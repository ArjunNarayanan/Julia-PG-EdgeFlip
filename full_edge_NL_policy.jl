using Flux
include("full_edge_policy_gradient.jl")
include("full_edge_model.jl")

PG = FullEdgePolicyGradient

struct FullEdgePolicyNL
    emodels
    lmodel::Any
    num_levels
    num_hidden_channels
    function FullEdgePolicyNL(num_levels, num_hidden_channels)
        emodels = [FullEdgeModel(4, num_hidden_channels)]
        for i in 2:num_levels
            push!(emodels, FullEdgeModel(num_hidden_channels, num_hidden_channels))
        end

        lmodel = Dense(16, 1)
        new(emodels, lmodel, num_levels, num_hidden_channels)
    end
end

function eval_single(p::FullEdgePolicyNL, ep, econn)
    x = p.emodels[1](ep, econn)
    x = relu.(x)

    for i in 2:length(p.emodels)
        y = p.emodels[i](x, econn)
        y = x + y
        x = relu.(y)
    end

    logits = p.lmodel(x)
    return logits
end

function eval_batch(p::FullEdgePolicyNL, ep, econn)
    nf, na, nb = size(ep)

    ep = reshape(ep, nf, na*nb)
    ep = eval_single(p, ep, econn)
    ep = reshape(ep, :, na, nb)
    return ep
end

function (p::FullEdgePolicyNL)(ep, econn)
    d = ndims(ep)
    if d == 2
        return eval_single(p, ep, econn)
    elseif d == 3
        return eval_batch(p, ep, econn)
    else
        error("Expected ndims = 2,3 got ndims = $d")
    end
end

Flux.@functor FullEdgePolicyNL

function Base.show(io::IO, policy::FullEdgePolicyNL)
    s = "PolicyNL\n\t $(policy.num_levels) levels\n\t $(policy.num_hidden_channels) channels"
    println(io, s)
end