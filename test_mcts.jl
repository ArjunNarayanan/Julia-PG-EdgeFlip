using OrderedCollections
using EdgeFlip

include("MCTS.jl")
include("policy_and_value_network.jl")

TS = MCTS
PV = PolicyAndValueNetwork

function TS.state(env::EdgeFlip.OrderedGameEnv)
    ets = copy(EdgeFlip.edge_template_score(env))
    econn = copy(EdgeFlip.edge_connectivity(env))
    epairs = copy(EdgeFlip.edge_pairs(env))

    idx = findall(epairs .== 0)
    epairs[idx] .= idx

    return ets, econn, epairs
end

function TS.step!(env::EdgeFlip.OrderedGameEnv, action)
    triangle, vertex = action
    EdgeFlip.step!(env, triangle, vertex)
end

function TS.reset!(env::EdgeFlip.OrderedGameEnv; nflips = 11, maxflipfactor = 1.0)
    maxflips = ceil(Int, maxflipfactor * nflips)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end


nref = 1
env = EdgeFlip.OrderedGameEnv(nref, 0)
TS.reset!(env)
pvnet = PV.PVNet(3, 16)

ets, econn, epairs = TS.state(env)