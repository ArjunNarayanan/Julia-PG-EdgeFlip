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
    ep = reshape(ep, nf, :)
    ep = ep[:, econn]

    ep = reshape(ep, 6nf, na*nb)
    ep = em.model(ep)

    ep = em.batchnorm(ep)

    ep = reshape(ep, :, na, nb)

    return ep
end

function (em::FullEdgeModel)(ep, econn)
    
    d = ndims(ep)

    if d == 2
        return eval_single(em, ep, econn)
    elseif d == 3
        return eval_batch(em, ep, econn)
    else
        error("Expected d == 2, 3 got d == $d")
    end

    return ep
end
