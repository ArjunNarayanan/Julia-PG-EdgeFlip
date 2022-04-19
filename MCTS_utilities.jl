include("MCTS.jl")
TS = MCTS

struct StateData
    edge_template_score::Any
    edge_connectivity::Any
    edge_pairs::Any
    normalized_remaining_flips::Any
    function StateData(edge_connectivity)
        edge_template_score = Vector{Matrix{Int}}(undef, 0)
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

function action_to_edgeix(action)
    triangle, vertex = div(action - 1, 3) + 1, (action - 1) % 3 + 1
    return triangle, vertex
end

function TS.step!(env::EdgeFlip.OrderedGameEnv, action; no_flip_reward = 0)
    triangle, vertex = action_to_edgeix(action)
    EdgeFlip.step!(env, triangle, vertex, no_flip_reward = no_flip_reward)
end

function TS.reverse_step!(env::EdgeFlip.OrderedGameEnv, action)
    triangle, vertex = action_to_edgeix(action)
    EdgeFlip.reverse_step!(env, triangle, vertex)
end

function TS.reset!(env::EdgeFlip.OrderedGameEnv; nflips = 10, maxflipfactor = 1.0)
    maxflips = ceil(Int, maxflipfactor * nflips)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end

function TS.is_terminal(env::EdgeFlip.OrderedGameEnv)
    EdgeFlip.done(env)
end

function TS.reward(env)
    EdgeFlip.normalized_reward(env)
end

function TS.action_probabilities_and_value(policy, state)
    ets, econn, epairs, normalized_remaining_flips = state
    p, v = PV.eval_single(policy, ets, econn, epairs, normalized_remaining_flips)
    return p, v
end
