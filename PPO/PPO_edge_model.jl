struct EdgeModel
    emodel
    pmodel
    batchnorm
    function EdgeModel(in_channels, out_channels)
        # emodel = Chain(Dense(3in_channels, out_channels, relu),
        #                Dense(out_channels, out_channels, relu))
        # pmodel = Chain(Dense(2out_channels, out_channels, relu),
        #                Dense(out_channels, out_channels))
        emodel = Dense(3in_channels, out_channels)
        pmodel = Dense(2out_channels, out_channels)
        batchnorm = BatchNorm(out_channels)
        new(emodel, pmodel, batchnorm)
    end
end

Flux.@functor EdgeModel

function cycle_edges(ep)
    nf, na = size(ep)
    ep = reshape(ep, nf, 3, :)

    ep1 = reshape(ep, 3nf, 1, :)
    ep2 = reshape(ep[:,[2,3,1],:], 3nf, 1, :)
    ep3 = reshape(ep[:,[3,1,2],:], 3nf, 1, :)

    ep = reshape(cat(ep1, ep2, ep3, dims = 2), 3nf, :)

    return ep
end

function eval_single(m::EdgeModel, ep, epairs, bvals)
    ep = cycle_edges(ep)
    ep = m.emodel(ep)

    epb = cat(ep,bvals,dims=2)
    ep2 = epb[:,epairs]

    ep = cat(ep,ep2,dims=1)
    ep = m.pmodel(ep)

    ep = m.batchnorm(ep)

    return ep
end