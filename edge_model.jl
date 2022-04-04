struct EdgeModel
    model::Any
    bvals::Any
    batchnorm::Any
    function EdgeModel(in_channels, out_channels)
        model = Dense(3in_channels, out_channels)
        bvals = Flux.glorot_uniform(in_channels)
        batchnorm = BatchNorm(out_channels)
        new(model, bvals, batchnorm)
    end
end

Flux.@functor EdgeModel

function eval_single(em::EdgeModel, ep, econn, epairs)
    nf, na = size(ep)

    ep = cat(ep, em.bvals, dims = 2)
    ep = reshape(ep[:, econn], 3nf, na)
    ep = em.model(ep)

    ep2 = ep[:, epairs]
    ep = 0.5 * (ep + ep2)

    ep = em.batchnorm(ep)

    return ep
end

function eval_batch(em::EdgeModel, ep, econn, epairs)
    nf, na, nb = size(ep)

    ep = cat(ep, repeat(em.bvals, inner = (1, 1, nb)), dims = 2)
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
