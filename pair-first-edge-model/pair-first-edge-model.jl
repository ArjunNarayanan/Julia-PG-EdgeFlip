struct PairEdgeModel
    pmodel::Any
    emodel::Any
    batchnorm::Any
    function PairEdgeModel(in_channels, hidden_channels, out_channels)
        pmodel = Chain(
            Dense(2in_channels, hidden_channels, relu),
            Dense(hidden_channels, hidden_channels, relu),
        )
        emodel = Chain(
            Dense(3hidden_channels, hidden_channels, relu),
            Dense(hidden_channels, out_channels),
        )
        batchnorm = BatchNorm(out_channels)

        new(pmodel, emodel, batchnorm)
    end
end

Flux.@functor PairEdgeModel

function eval_single(p::PairEdgeModel, ep, econn, epairs)

    x = ep[:, 1:end-1]

    ep2 = ep[:, epairs]
    ep = cat(x, ep2, dims = 1)
    ep = p.pmodel(ep)

    nf, na = size(ep)
    ep = reshape(ep[:, econn], 3nf, na)

    ep = p.emodel(ep)

    ep = p.batchnorm(ep)

    return ep
end

function eval_batch(p::PairEdgeModel, ep, econn, epairs)
    nf, na, nb = size(ep)

    x = ep[:, 1:end-1, :]
    x = reshape(x, nf, :)

    ep = reshape(ep, nf, na * nb)
    ep2 = ep[:, epairs]
    ep = cat(x, ep2, dims = 1)
    ep = p.pmodel(ep)

    nf, _ = size(ep)
    ep = reshape(ep, nf, na - 1, nb)
    ep = reshape(ep[:, econn, :], 3nf, na - 1, nb)
    ep = p.emodel(ep)

    nf, na, nb = size(ep)
    ep = reshape(ep, nf, :)

    ep = p.batchnorm(ep)

    ep = reshape(ep, nf, na, nb)
    
    return ep
end
