module Value

using Flux
include("PPO_edge_model.jl")

struct ValueNL
    emodels
    bvals
    vmodel
    num_levels
    num_hidden_channels
    function ValueNL(num_levels, num_hidden_channels)
        emodels = [EdgeModel(4, num_hidden_channels)]
        for i = 2:num_levels
            push!(emodels, EdgeModel(num_hidden_channels, num_hidden_channels))
        end

        bvals = Flux.glorot_uniform(num_hidden_channels)
        vmodel = Chain(Dense(num_hidden_channels + 1, num_hidden_channels, relu),
                       Dense(num_hidden_channels, 1))
        new(emodels, bvals, vmodel, num_levels, num_hidden_channels)
    end
end

Flux.@functor ValueNL

function Base.show(io::IO, value::ValueNL)
    s = "ValueNL\n\t $(value.num_levels) levels\n\t $(value.num_hidden_channels) channels"
    println(io, s)
end

function final_activations(v::ValueNL, ep, epairs)
    x = eval_single(v.emodels[1], ep, epairs, p.bvals)
    x = relu.(x)

    for i in 2:length(v.emodels)
        y = eval_single(v.emodels[i], x, epairs, p.bvals)
        y = x + y
        x = relu.(y)
    end

    return x
end

function eval_single(v::ValueNL, ep, epairs, normalized_remaining_flips)
    x = final_activations(v, ep, epairs)
    
    mean_x = Flux.mean(x, dims = 2)
    x = vcat(mean_x, normalized_remaining_flips)
    val = first(v.vmodel(x))

    return val
end

function eval_batch(v::ValueNL, ep, epairs, normalized_remaining_flips)
    nf, na, nb = size(ep)
    ep = reshape(ep, nf, :)
    epairs = vec(epairs)

    x = final_activations(v, ep, epairs)
    nf, _ = size(x)
    x = reshape(x, nf, na, nb)
    mean_x = Flux.mean(x, dims = 2)

    x = vcat(mean_x, normalized_remaining_flips')
    vals = vec(p.vmodel(x))

    return vals
end


end