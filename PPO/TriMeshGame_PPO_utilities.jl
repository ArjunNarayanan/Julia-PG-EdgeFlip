using Flux
using Distributions: Categorical
using TriMeshGame
include("PPO.jl")
include("PPO_split_policy.jl")


TM = TriMeshGame

function val_or_zero(vector,template)
    return [t == 0 ? 0 : vector[t] for t in template]
end

function PPO.state(env)
    vs = val_or_zero(env.vertex_score, env.template)
    v0 = val_or_zero(env.d0, env.template)

    return vcat(vs,v0)
end

function PPO.is_terminal(env)
    return env.is_terminated
end

function PPO.reward(env)
    return env.reward
end

function index_to_action(index)
    triangle = div(index - 1, 6)+1
    vertex = rem(index-1,3)+1
    pos = rem(index - 1, 6)
    type = div(pos,3)+1

    return triangle, vertex, type
end

function action_space_size(env)
    return size(env.template,2)*2
end

function PPO.step!(env, action_index; no_action_reward=-4)
    na = action_space_size(env)
    @assert 0 < action_index <= na "Expected 0 < action_index <= $na, got action_index = $action_index"
    @assert !env.is_terminated "Attempting to step in terminated environment with action $action_index"
    triangle, vertex, type = index_to_action(action_index)
    TM.step!(env, triangle, vertex, type, no_action_reward=no_action_reward)
end

struct StateData
    state
end

function StateData()
    state = Matrix{Int}[]
    StateData(state)
end

function Base.length(s::StateData)
    return length(s.state)
end

function Base.show(io::IO, s::StateData)
    l = length(s)
    println(io, "StateData")
    println(io, "\t$l data points")
end

function PPO.update!(state_data::StateData, state)
    push!(state_data.state, state)
end

function PPO.initialize_state_data(env)
    return StateData()
end

function PPO.action_probabilities(policy, state)
    logits = vec(policy(state))
    p = softmax(logits)
    return p
end

