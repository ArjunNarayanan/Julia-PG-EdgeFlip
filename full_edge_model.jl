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

function (em::FullEdgeModel)(ep, econn)
    nf, na = size(ep)

    ep = cat(ep, em.bvals, dims = 2)
    ep = reshape(ep[:, econn], 6nf, na)

    ep = em.model(ep)
    ep = em.batchnorm(ep)

    return ep
end

Flux.@functor FullEdgeModel