using EdgeFlip
using BSON
using Flux
include("circlemesh.jl")
include("../NL_policy.jl")

PG = EdgePolicyGradient

function PG.state(env::EdgeFlip.OrderedGameEnv)
    ets = copy(EdgeFlip.edge_template_score(env))
    econn = copy(EdgeFlip.edge_connectivity(env))
    epairs = copy(EdgeFlip.edge_pairs(env))

    idx = findall(epairs .== 0)
    epairs[idx] .= idx

    return ets, econn, epairs
end

function PG.step!(env::EdgeFlip.OrderedGameEnv, action)
    triangle, vertex = action
    EdgeFlip.step!(env, triangle, vertex)
end

function PG.is_terminated(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.is_terminated(env)
end

function PG.reward(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.reward(env)
end

function PG.reset!(env::EdgeFlip.OrderedGameEnv; nflips = 0, maxflips = EdgeFlip.number_of_actions(env))
    EdgeFlip.reset!(env, maxflips = maxflips)
end

function PG.score(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.score(env)
end


# element_size = 0.2
# maxflipfactor = 1.0
# filename = "results/models/3L-model/policy-5500.bson"
# BSON.@load filename policy new_rets
# env = circle_ordered_game_env(element_size, maxflipfactor = maxflipfactor)

PG.reset!(env)
nn_ret = PG.normalized_returns(env, policy, 100)

optimizer = ADAM(1e-4)
epochs, rets = PG.run_training_loop(env, policy, optimizer, 2000, 1, 100, print_every = 10)