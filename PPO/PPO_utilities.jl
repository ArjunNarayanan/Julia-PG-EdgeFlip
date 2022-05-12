using Flux
using Distributions: Categorical
using EdgeFlip
include("PPO.jl")

include("PPO_direct_policy.jl")

# include("PPO_NL_policy.jl")
# include("PPO_NL_value.jl")

# include("simple_edge_policy.jl")
# Policy = SimplePolicy

EF = EdgeFlip

function PPO.state(env)
    # ets = copy(EF.edge_template_score(env))
    ets = EF.edge_template_score(env)[[1],:]

    epairs = copy(EF.edge_pairs(env))
    normalized_remaining_flips = (env.maxflips - env.nbrflips)/EdgeFlip.number_of_actions(env)
    remaining_score = EF.score(env) - EF.optimum_score(env)

    return ets, epairs, normalized_remaining_flips, remaining_score
end

function PPO.is_terminal(env)
    return EF.done(env)
end

function PPO.reward(env)
    return EF.reward(env)
end

function PPO.reset!(env)
    EF.reset!(env)
end

function action_to_edgeix(action)
    triangle, vertex = div(action - 1, 3) + 1, (action - 1) % 3 + 1
    return triangle, vertex
end

function PPO.step!(env, action; no_flip_reward = -4)
    na = EdgeFlip.number_of_actions(env)
    @assert 0 < action <= na "Expected 0 < action <= $na got action = $action"
    @assert !EdgeFlip.done(env) "Attempting to step in done environment with action $action"
    triangle, vertex = action_to_edgeix(action)
    EdgeFlip.step!(env, triangle, vertex, no_flip_reward = no_flip_reward)
end

struct StateData
    edge_template_score::Any
    edge_pairs::Any
    remaining_flips
    remaining_score
end

function StateData(env)
    edge_template_score = Matrix{Int}[]
    edge_pairs = Vector{Int}[]
    remaining_flips = Float64[]
    remaining_score = Int[]
    StateData(
        edge_template_score,
        edge_pairs,
        remaining_flips,
        remaining_score
    )
end

function Base.length(s::StateData)
    return last(size(s.edge_template_score))
end

function Base.show(io::IO, s::StateData)
    l = length(s)
    println(io, "StateData")
    println(io, "\t$l data points")
end

function PPO.update!(state_data::StateData, state)
    ets, epairs, remaining_flips, remaining_score = state
    push!(state_data.edge_template_score, ets)
    push!(state_data.edge_pairs, epairs)
    push!(state_data.remaining_flips, remaining_flips)
    push!(state_data.remaining_score, remaining_score)
end

function remaining_score(state_data::StateData)
    return state_data.remaining_score
end

function PPO.initialize_state_data(env)
    return StateData(env)
end

function batch_offset_edge_pairs!(epairs)
    na, nb = size(epairs)
    zero_index = findall(epairs .== 0)

    for (idx, col) in enumerate(eachcol(epairs))
        col .+= (idx - 1) * na
    end

    # rows = [z[1] for z in zero_index]
    # epairs[zero_index] .= rows

    epairs[zero_index] .= (na*nb + 1)
end

function offset_edge_pairs!(epairs)
    offset = length(epairs) + 1
    epairs[epairs .== 0] .= offset

    # idx = findall(epairs .== 0)
    # epairs[idx] .= idx
end

function PPO.episode_state(state_data)
    ets = cat(state_data.edge_template_score..., dims = 3)
    epairs = cat(state_data.edge_pairs..., dims = 2)

    return StateData(ets, epairs, state_data.remaining_flips, state_data.remaining_score)
end

function PPO.batch_state(vec_state_data)
    ets = cat([state_data.edge_template_score for state_data in vec_state_data]..., dims = 3)
    
    epairs = cat([state_data.edge_pairs for state_data in vec_state_data]..., dims = 2)
    batch_offset_edge_pairs!(epairs)

    nflips = cat([state_data.remaining_flips for state_data in vec_state_data]..., dims = 1)

    return ets, epairs, nflips
end

function PPO.action_probabilities(policy, state)
    ets, epairs, _, _ = state
    pairs = copy(epairs)

    offset_edge_pairs!(pairs)
    logits = Policy.eval_single(policy, ets, pairs)

    p = softmax(logits)

    return p
end

function PPO.episode_returns(rewards, state_data, discount)
    ne = length(rewards)

    values = zeros(ne)
    v = 0.0

    for idx = ne:-1:1
        v = rewards[idx] + discount * v
        values[idx] = v
    end

    normalized_values = values ./ remaining_score(state_data)
    return normalized_values
end

function PPO.batch_action_probabilities(policy, state)
    ets, epairs, _ = state
    logits = Policy.eval_batch(policy, ets, epairs)

    p = softmax(logits, dims = 1)

    return p
end

function single_trajectory_return(env, policy)
    done = PPO.is_terminal(env)
    if done
        return 0.0
    else
        initial_score = EF.score(env)
        minscore = initial_score
        while !done
            probs = PPO.action_probabilities(policy, PPO.state(env))
            action = rand(Categorical(probs))

            PPO.step!(env, action)

            minscore = min(minscore, EF.score(env))
            done = PPO.is_terminal(env)
        end
        return initial_score - minscore
    end
end

function single_trajectory_normalized_return(env, policy)
    maxreturn = EF.score(env) - env.optimum_score
    if maxreturn == 0
        return 1.0
    else
        ret = single_trajectory_return(env, policy)
        return ret / maxreturn
    end
end

function average_normalized_returns(env, policy, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(env)
        ret[idx] = single_trajectory_normalized_return(env, policy)
    end
    return Flux.mean(ret), Flux.std(ret)
end
