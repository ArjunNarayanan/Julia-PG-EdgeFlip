module Policy

using Flux

struct DirectPolicy
    model
    bval1
    bval2
    bval3
    hidden_channels
    num_hidden_layers
    function DirectPolicy(in_channels, hidden_channels, num_hidden_layers)
        model = []
        push!(model, Dense(in_channels, hidden_channels, relu))
        for i in 1:num_hidden_layers-1
            push!(model, Dense(hidden_channels, hidden_channels, relu))
        end
        push!(model, Dense(hidden_channels, 1))
        model = Chain(model...)

        bval1 = Flux.glorot_uniform(1)
        bval2 = Flux.glorot_uniform(2)
        bval3 = Flux.glorot_uniform(4)

        new(model, bval1, bval2, bval3, hidden_channels, num_hidden_layers)
    end
end

Flux.@functor DirectPolicy

function Base.show(io::IO, p::DirectPolicy)
    s = "DirectPolicy\n\t$(p.hidden_channels) channels\n\t$(p.num_hidden_layers) layers"
    println(io, s)
end

function cycle_edges(x)
    nf, na = size(x)
    x = reshape(x, nf, 3, :)

    x1 = reshape(x, 3nf, 1, :)
    x2 = reshape(x[:,[2,3,1],:], 3nf, 1, :)
    x3 = reshape(x[:,[3,1,2],:], 3nf, 1, :)

    x = reshape(cat(x1, x2, x3, dims = 2), 3nf, :)
    return x
end

function get_pairs(x, pairs, bval)
    x = cat(x, bval, dims = 2)
    return x[:, pairs]
end

function eval_single(m::DirectPolicy, x, pairs)
    cx = cycle_edges(x)
    
    px = get_pairs(x, pairs, m.bval1)
    cpx = cycle_edges(px)

    y = cpx[[2,3], :]
    py = get_pairs(y, pairs, m.bval2)
    cpy = cycle_edges(py)

    z = cpy[3:6, :]
    pz = get_pairs(z, pairs, m.bval3)

    model_input = cat(cx, cpx, cpy, pz, dims = 1)

    model_output = vec(m.model(model_input))

    return model_output
end

function eval_batch(m::DirectPolicy, x, pairs)
    nf, na, nb = size(x)
    x = reshape(x, nf, :)
    pairs = vec(pairs)

    logits = eval_single(m, x, pairs)
    logits = reshape(logits, na, nb)

    return logits
end

end