module SimplePolicy

using Flux

struct SPolicy
    model
    bvals
    num_gather_ops
    num_hidden_layers
    hidden_layer_width
    function SPolicy(num_gather_ops, num_hidden_layers, hidden_layer_width)
        num_features = 2*6^num_gather_ops

        model = []
        model = push!(model, Dense(num_features, hidden_layer_width, relu))
        for i in 1:num_hidden_layers
            push!(model, Dense(hidden_layer_width, hidden_layer_width, relu))
        end
        push!(model, Dense(hidden_layer_width, 1))

        model = Chain(model...)

        bvals = [Flux.glorot_uniform(2*6^i) for i in 0:num_gather_ops-1]

        new(model, bvals, num_gather_ops, num_hidden_layers, hidden_layer_width)
    end
end

Flux.@functor SPolicy

function Base.show(io::IO, policy::SPolicy)
    s = "SPolicy\n\t $(policy.num_gather_ops) gather steps\n\t $(policy.num_hidden_layers) hidden layers\n\t $(policy.hidden_layer_width) width"
    println(io, s)
end

function gather_pairs(x, pairs, bval)
    xb = cat(x, bval, dims = 2)
    xp = xb[:,pairs]
    return cat(x, xp, dims = 1)
end

function gather_neighbors(x)
    nf, na = size(x)
    x = reshape(x, nf, 3, :)

    x1 = reshape(x, 3nf, 1, :)
    x2 = reshape(x[:,[2,3,1],:], 3nf, 1, :)
    x3 = reshape(x[:,[3,1,2],:], 3nf, 1, :)

    x = cat(x1, x2, x3, dims = 2)
    x = reshape(x, 3nf, :)

    return x
end

function gather(x, pairs, bvals, numsteps)
    for i = 1:numsteps
        x = gather_pairs(x, pairs, bvals[i])
        x = gather_neighbors(x)
    end
    return x
end

function (p::SPolicy)(x, pairs)
    x = gather(x, pairs, p.bvals, p.num_gather_ops)
    return p.model(x)
end

end