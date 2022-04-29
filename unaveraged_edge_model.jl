struct UnaveragedEdgeModel
    tmodel::Any
    nmodel::Any
    batchnorm::Any
    function UnaveragedEdgeModel(in_channels, out_channels)
        tmodel = Dense(3in_channels, out_channels)
        nmodel = Dense(2out_channels, out_channels)
        batchnorm = BatchNorm(out_channels)

        new(model, batchnorm)
    end
end

Flux.@functor UnaveragedEdgeModel

function evaluate_edge_model(model, ep)

    slice1 = reshape(ep[:,1,:], nf, 1, :)
    slice2 = reshape(ep[:,2,:], nf, 1, :)
    slice3 = reshape(ep[:,3,:], nf, 1, :)
    
    col1 = cat(slice1, slice2, slice3, dims = 1)
    col2 = cat(slice2, slice3, slice1, dims = 1)
    col3 = cat(slice3, slice1, slice2, dims = 1)

    ep = cat(col1, col2, col3, dims = 2)
    ep = model(ep)

    return ep
end

function eval_single(em::UnaveragedEdgeModel, ep, epairs)
    nf, na = size(ep)

    ep = reshape(ep, nf, 3, :)
    ep = evaluate_edge_model(em.tmodel, ep)

    nf, _, _ = size(ep)
    ep = reshape(ep, nf, na)

    ep2 = ep[:, epairs]
    ep = vcat(ep, ep2)
    ep = em.nmodel(ep)

    ep = em.batchnorm(ep)

    return ep
end

function eval_batch(em::UnaveragedEdgeModel, ep, epairs)
    nf, na, nb = size(ep)

    ep = reshape(ep, nf, 3, :)
    ep = evaluate_edge_model(em.tmodel, ep)

    nf, _, _ = size(ep)
    ep = reshape(ep, nf, :)

    ep2 = ep[:, epairs]
    ep = vcat(ep, ep2)
    ep = em.nmodel(ep)
    
    ep = em.batchnorm(ep)

    nf, _ = size(ep)
    ep = reshape(ep, nf, na, nb)

    return ep
end