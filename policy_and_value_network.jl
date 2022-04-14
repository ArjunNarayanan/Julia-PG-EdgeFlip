module PolicyAndValueNetwork

using Flux
include("edge_model.jl")

struct PVNet
    emodels
    pmodel
    vmodel
    num_levels
    num_hidden_channels
    function PVNet(num_levels, num_hidden_channels)
        emodels = [EdgeModel(4, num_hidden_channels)]
        for i = 2:num_levels
            push!(emodels, EdgeModel(num_hidden_channels, num_hidden_channels))
        end
        pmodel = Chain(Dense(num_hidden_channels, num_hidden_channels, relu),
                       Dense(num_hidden_channels, 1))
        vmodel = Chain(Dense(num_hidden_channels, num_hidden_channels, relu),
                       Dense(num_hidden_channels, 1))

        new(emodels, pmodel, vmodel, num_levels, num_hidden_channels)
    end
end

Flux.@functor PVNet

function Base.show(io::IO, policy::PVNet)
    s = "PVNet\n\t $(policy.num_levels) levels\n\t $(policy.num_hidden_channels) channels"
    println(io, s)
end

function eval_single(p::PVNet, ep, econn, epairs)
    x = eval_single(p.emodels[1], ep, econn, epairs)
    x = relu.(x)

    for i in 2:p.num_levels
        y = eval_single(p.emodels[i], x, econn, epairs)
        y = x + y
        x = relu.(y)
    end

    probs = softmax(vec(p.pmodel(x)))
    
    val = sum(p.vmodel(x))

    return probs, val
end

end