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

function eval_single(em::EdgeModel, ep, econn, epairs)
    nf, na = size(ep)

    ep = reshape(ep[:, econn], 3nf, na)
    ep = em.model(ep)

    ep2 = ep[:, epairs]
    ep = 0.5 * (ep + ep2)

    ep = em.batchnorm(ep)

    return ep
end

function eval_batch(em::EdgeModel, ep, econn, epairs)
    nf, na, nb = size(ep)

    ep = reshape(ep[:, econn, :], 3nf, na, nb)
    ep = em.model(ep)

    nf, na, nb = size(ep)
    ep = reshape(ep, nf, :)
    ep2 = ep[:, epairs]
    ep = 0.5*(ep + ep2)

    ep = em.batchnorm(ep)

    ep = reshape(ep, nf, na, nb)
    return ep
end
