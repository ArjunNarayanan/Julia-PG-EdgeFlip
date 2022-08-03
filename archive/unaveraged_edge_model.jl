struct UnaveragedEdgeModel
    tmodel::Any
    nmodel::Any
    batchnorm::Any
    function UnaveragedEdgeModel(in_channels, out_channels)
        tmodel = Dense(3in_channels, out_channels)
        nmodel = Dense(2out_channels, out_channels)
        batchnorm = BatchNorm(out_channels)

        new(tmodel, nmodel, batchnorm)
    end
end

Flux.@functor UnaveragedEdgeModel

function eval_single(em::UnaveragedEdgeModel, ep, econn, epairs)
    nf, na = size(ep)

    ep = reshape(ep[:, econn], 3nf, na)
    ep = em.tmodel(ep)

    ep2 = ep[:, epairs]
    ep = vcat(ep, ep2)
    ep = em.nmodel(ep)

    ep = em.batchnorm(ep)

    return ep
end

function eval_batch(em::UnaveragedEdgeModel, ep, econn, epairs)
    nf, na, nb = size(ep)

    ep = reshape(ep[:, econn, :], 3nf, na, nb)
    ep = em.tmodel(ep)

    nf, na, nb = size(ep)
    ep = reshape(ep, nf, na*nb)

    ep2 = ep[:, epairs]
    ep = vcat(ep, ep2)
    ep = em.nmodel(ep)
    
    ep = em.batchnorm(ep)

    ep = reshape(ep, nf, na, nb)

    return ep
end