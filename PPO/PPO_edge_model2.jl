struct EdgeModel
    model::Any
    batchnorm::Any
    function EdgeModel(in_channels, out_channels)
        model = Dense(3in_channels, out_channels)
        batchnorm = BatchNorm(out_channels)
        new(model, batchnorm)
    end
end

Flux.@functor EdgeModel

function cycle_edges(ep)
    nf, na = size(ep)
    ep = reshape(ep, nf, 3, :)

    ep1 = reshape(ep, 3nf, 1, :)
    ep2 = reshape(ep[:, [2, 3, 1], :], 3nf, 1, :)
    ep3 = reshape(ep[:, [3, 1, 2], :], 3nf, 1, :)

    ep = reshape(cat(ep1, ep2, ep3, dims = 2), 3nf, :)

    return ep
end

function eval_single(m::EdgeModel, ep, epairs, bvals)
    ep = cycle_edges(ep)
    ep = m.model(ep)

    epb = cat(ep, bvals, dims = 2)
    ep2 = epb[:, epairs]

    ep = 0.5*(ep + ep2)

    ep = m.batchnorm(ep)

    return ep
end