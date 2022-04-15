using OrderedCollections
using EdgeFlip
using Distributions: Categorical

include("MCTS.jl")
include("policy_and_value_network.jl")

TS = MCTS
PV = PolicyAndValueNetwork

struct StateData
    edge_template_score::Any
    edge_connectivity::Any
    edge_pairs::Any
    normalized_remaining_flips::Any
    function StateData()
        edge_template_score = Vector{Matrix{Int}}(undef, 0)
        edge_connectivity = Vector{Vector{Int}}(undef, 0)
        edge_pairs = Vector{Vector{Int}}(undef, 0)
        normalized_remaining_flips = Float64[]
        new(edge_template_score, edge_connectivity, edge_pairs, normalized_remaining_flips)
    end
end

function Base.length(s::StateData)
    return length(s.edge_template_score)
end

function Base.show(io::IO, s::StateData)
    l = length(s)
    println(io, "StateData")
    println(io, "\t$l data points")
end

function TS.update!(state_data::StateData, state)
    ets, econn, epairs, nflips = state
    push!(state_data.edge_template_score, ets)
    push!(state_data.edge_connectivity, econn)
    push!(state_data.edge_pairs, epairs)
    push!(state_data.normalized_remaining_flips, nflips)
end

function TS.state(env::EdgeFlip.OrderedGameEnv)
    ets = copy(EdgeFlip.edge_template_score(env))
    econn = copy(EdgeFlip.edge_connectivity(env))
    epairs = copy(EdgeFlip.edge_pairs(env))

    idx = findall(epairs .== 0)
    epairs[idx] .= idx

    normalized_remaining_flips =
        (env.maxflips - env.nbrflips) / EdgeFlip.number_of_actions(env)

    return ets, econn, epairs, normalized_remaining_flips
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
    ets, econn, epairs, normalized_remaining_flips = state
    p, v = PV.eval_single(policy, ets, econn, epairs, normalized_remaining_flips)
    return p, v
end

function TS.is_terminal(env::EdgeFlip.OrderedGameEnv)
    EdgeFlip.done(env)
end

function TS.reward(env)
    r = EdgeFlip.reward(env)
    optimum = r + EdgeFlip.score(env) - EdgeFlip.optimum_score(env)
    return r / optimum
end

nref = 1
env = EdgeFlip.OrderedGameEnv(nref, 0)
policy = PV.PVNet(3, 16)

Cpuct = 1.0
temperature = 1.0
discount = 1.0
maxtime = 0.1

TS.reset!(env)

p, v = TS.action_probabilities_and_value(policy, TS.state(env))
root = TS.Node(p, TS.is_terminal(env))
root = TS.search(root, env, policy, Cpuct, discount, maxtime)

# batch_data = TS.BatchData(StateData())
# TS.step_mcts!(batch_data, env, policy, Cpuct, discount, maxtime, temperature)

# root = TS.search(env, policy, Cpuct, discount, maxtime)
# ap = TS.mcts_action_probabilities(root.visit_count, 72, 1.0)



# p = Categorical(TS.mcts_action_probabilities(root.visit_count, 72, temperature))


# n2, a2 = TS.select(root, env, 1)

# p, v = TS.action_probabilities_and_value(policy, TS.state(env))
# root = TS.Node(p)
# node, action = TS.select(root, env, 1)
# child, val = TS.expand(node, action, env, policy)
# root = TS.backup(child, val, 1.0, env)

# node, action = TS.select(root, env, 1)
# child, val = TS.expand(node, action, env, policy)