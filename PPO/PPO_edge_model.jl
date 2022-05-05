struct EdgeModel
    pmodel
    emodel
    batchnorm
    function EdgeModel(in_channels, out_channels)
        pmodel = Dense(2in_channels, out_channels, relu)
        emodel = Dense(3out_channels, out_channels)
        batchnorm = BatchNorm(out_channels)
        new(pmodel, emodel, batchnorm)
    end
end

Flux.@functor EdgeModel

function eval_single(m::EdgeModel, ep, epairs, bvals)
    epb = cat(ep, bvals, dims = 2)
    ep2 = epb[:, epairs]

    x = cat(ep, ep2, dims = 1)
    x = m.pmodel(x)
end