using Flux
using Distributions: Categorical
using EdgeFlip
include("PPO.jl")
include("PPO_NL_policy.jl")

EF = EdgeFlip

function PPO.state(env)
    ets = copy(EF.edge_template_score(env))
    epairs = copy(EF.edge_pairs(env))
    
    return ets, epairs
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
end

function StateData(env)
    edge_template_score = Matrix{Int}[]
    edge_pairs = Vector{Int}[]
    StateData(
        edge_template_score,
        edge_pairs,
    )
end

function Base.length(s::StateData)
    @assert length(s.edge_template_score) == length(s.edge_pairs)
    return length(s.edge_template_score)
end

function Base.show(io::IO, s::StateData)
    l = length(s)
    println(io, "StateData")
    println(io, "\t$l data points")
end

function PPO.update!(state_data::StateData, state)
    ets, epairs = state
    push!(state_data.edge_template_score, ets)
    push!(state_data.edge_pairs, epairs)
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

    epairs[zero_index] .= (na*nb + 1)
end

function offset_edge_pairs(epairs)
    offset = length(epairs) + 1
    offset_epairs = [p == 0 ? offset : p for p in epairs]
    return offset_epairs
end

function PPO.batch_state(state_data)
    ets = cat(state_data.edge_template_score..., dims = 3)

    epairs = cat(state_data.edge_pairs..., dims = 2)
    batch_offset_edge_pairs!(epairs)
    epairs = vec(epairs)

    return ets, epairs
end

function PPO.action_probabilities(policy, state)
    ets, epairs = state
    epairs = offset_edge_pairs(epairs)
    logits = Policy.eval_single(policy, ets, epairs)

    p = softmax(logits)

    return p
end

function PPO.batch_action_probabilities(policy, state)
    ets, epairs = state
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
    return mean(ret), std(ret)
end
