using Flux
include("pair-first-policy-gradient.jl")
include("pair-first-edge-model.jl")

PG = EdgePolicyGradient

struct PairPolicy
    emodels
    lmodel
    bvals
    hbvals
    num_levels
    num_hidden_channels
    function PairPolicy(num_levels, num_hidden_channels)
        emodels = [PairEdgeModel(4, num_hidden_channels, num_hidden_channels)]
        for i in 2:num_levels
            push!(emodels, PairEdgeModel(num_hidden_channels, num_hidden_channels, num_hidden_channels))
        end

        lmodel = Dense(num_hidden_channels, 1)
        bvals = Flux.glorot_uniform(4)
        hbvals = Flux.glorot_uniform(num_hidden_channels)

        new(emodels, lmodel, bvals, hbvals, num_levels, num_hidden_channels)
    end
end

Flux.@functor PairPolicy

function Base.show(io::IO, policy::PairPolicy)
    s = "PairPolicy\n\t $(policy.num_levels) levels\n\t $(policy.num_hidden_channels) channels"
    println(io, s)
end

function PG.eval_single(p::PairPolicy, ep, econn, epairs)
    ep = cat(ep, p.bvals, dims = 2)
    x = eval_single(p.emodels[1], ep, econn, epairs)
    x = relu.(x)

    for i in 2:length(p.emodels)
        xb = cat(x, p.hbvals, dims = 2)
        y = eval_single(p.emodels[i], xb, econn, epairs)
        y = x + y
        x = relu.(y)
    end

    logits = p.lmodel(x)
    return logits
end

function PG.eval_batch(p::PairPolicy, ep, econn, epairs)
    nf, na, nb = size(ep)

    ep = cat(ep, repeat(p.bvals, inner=(1,1,nb)), dims = 2)
    x = eval_batch(p.emodels[1], ep, econn, epairs)
    x = relu.(x)

    for i = 2:length(p.emodels)
        xb = cat(x, repeat(p.hbvals, inner = (1,1,nb)), dims = 2)
        y = eval_batch(p.emodels[i], xb, econn, epairs)
        y = x + y
        x = relu.(y)
    end

    logits = p.lmodel(x)
    return logits
end