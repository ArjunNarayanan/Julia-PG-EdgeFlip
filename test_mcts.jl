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
    triangle, vertex = div(action - 1, 3) + 1, (action - 1) % 3 + 1
    EdgeFlip.step!(env, triangle, vertex)
end

function TS.reverse_step!(env::EdgeFlip.OrderedGameEnv, action)
    triangle, vertex = div(action - 1, 3) + 1, (action - 1) % 3 + 1
    EdgeFlip.reverse_step!(env, triangle, vertex)
end

function TS.reset!(env::EdgeFlip.OrderedGameEnv; nflips = 11, maxflipfactor = 1.0)
    maxflips = ceil(Int, maxflipfactor * nflips)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end

function TS.action_probabilities_and_value(policy, state)
    ets, econn, epairs = state
    p, v = PV.eval_single(policy, ets, econn, epairs)
    return p, v
end

function TS.is_terminal(env)
    EdgeFlip.done(env)
end

function TS.reward(env)
    r = EdgeFlip.reward(env)
    optimum = r + EdgeFlip.score(env) - EdgeFlip.optimum_score(env)
    return r/optimum
end

nref = 1
env = EdgeFlip.OrderedGameEnv(nref, 0)
TS.reset!(env)

policy = PV.PVNet(3, 16)
p, v = TS.action_probabilities_and_value(policy, TS.state(env))

root = TS.Node(p)
node, action = TS.select(root, env, 1)
child, val = TS.expand(node, action, env, policy)