struct FullEdgeModel
    model::Any
    bvals::Any
    batchnorm::Any
    function FullEdgeModel(in_channels, out_channels)
        model = Dense(6in_channels, out_channels)
        bvals = Flux.glorot_uniform(in_channels)
        batchnorm = BatchNorm(out_channels)
        new(model, bvals, batchnorm)
    end
end

Flux.@functor FullEdgeModel

function eval_single(em::FullEdgeModel, ep, econn)
    nf, na = size(ep)

    ep = cat(ep, em.bvals, dims = 2)
    ep = reshape(ep[:, econn], 6nf, na)
    ep = em.model(ep)
    ep = em.batchnorm(ep)

    return ep
end

function eval_batch(em::FullEdgeModel, ep, econn)
    nf, na, nb = size(ep)

    ep = cat(ep, repeat(em.bvals, inner = (1, 1, nb)), dims = 2)
    ep = reshape(ep, nf, na*nb + 1)
    ep = ep[:, econn]
    ep = reshape(ep, 6nf, na*nb)
    ep = em.model(ep)
    ep = em.batchnorm(ep)
    
    nf, _ = size(ep)

    ep = reshape(ep, nf, na, nb)
    return ep
end
