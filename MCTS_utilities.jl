using EdgeFlip
using Flux
using Distributions: Categorical
using Statistics
using Printf

include("MCTS.jl")
TS = MCTS
EF = EdgeFlip

struct StateData
    edge_template_score::Any
    edge_connectivity::Any
    edge_pairs::Any
    normalized_remaining_flips::Any
end

function StateData(env)
    edge_connectivity = EdgeFlip.edge_connectivity(env)
    edge_template_score = Vector{Matrix{Int}}(undef, 0)
    edge_pairs = Vector{Vector{Int}}(undef, 0)
    normalized_remaining_flips = Float64[]
    StateData(
        edge_template_score,
        edge_connectivity,
        edge_pairs,
        normalized_remaining_flips,
    )
end

function Base.length(s::StateData)
    return length(s.edge_template_score)
end

function offset_edge_pairs!(epairs)
    na, nb = size(epairs)
    for (idx, col) in enumerate(eachcol(epairs))
        col .+= (idx - 1) * na
    end
end

function TS.batch_state(s::StateData)
    ets = cat(s.edge_template_score..., dims = 3)
    econn = s.edge_connectivity

    epairs = cat(s.edge_pairs..., dims = 2)
    offset_edge_pairs!(epairs)
    epairs = vec(epairs)

    nflips = s.normalized_remaining_flips

    return ets, econn, epairs, nflips
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

function TS.reorder(state_data::StateData, idx)
    @assert length(idx) == length(state_data)
    ets = state_data.edge_template_score[idx]
    epairs = state_data.edge_pairs[idx]
    remflips = state_data.normalized_remaining_flips[idx]
    sd = StateData(ets, state_data.edge_connectivity, epairs, remflips)
    return sd
end

function Base.getindex(state_data::StateData, start_stop)

    ets = cat(state_data.edge_template_score[start_stop]..., dims = 3)
    econn = state_data.edge_connectivity

    epairs = cat(state_data.edge_pairs[start_stop]..., dims = 2)
    offset_edge_pairs!(epairs)
    epairs = vec(epairs)

    nflips = state_data.normalized_remaining_flips[start_stop]

    return ets, econn, epairs, nflips
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
    na = EdgeFlip.number_of_actions(env)
    @assert 0 < action <= na "Expected 0 < action <= $na got action = $action"
    @assert !EdgeFlip.done(env) "Attempting to step in done environment"
    triangle, vertex = action_to_edgeix(action)
    EdgeFlip.step!(env, triangle, vertex, no_flip_reward = no_flip_reward)
end

function TS.reverse_step!(env::EdgeFlip.OrderedGameEnv, action)
    triangle, vertex = action_to_edgeix(action)
    EdgeFlip.reverse_step!(env, triangle, vertex)
end

function TS.reset!(
    env::EdgeFlip.OrderedGameEnv;
    nflips = env.num_initial_flips,
    maxflipfactor = 1.0,
)
    maxflips = ceil(Int, maxflipfactor * nflips)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end

function TS.is_terminal(env::EdgeFlip.OrderedGameEnv)
    EdgeFlip.done(env)
end

function TS.reward(env)
    if TS.is_terminal(env)
        r =
            (EdgeFlip.initial_score(env) - EdgeFlip.score(env)) /
            (EdgeFlip.initial_score(env) - EdgeFlip.optimum_score(env))
        return r
    else
        return 0.0
    end
end

function TS.action_probabilities_and_value(policy, state)
    ets, econn, epairs, normalized_remaining_flips = state
    logits, v = PV.eval_single(policy, ets, econn, epairs, normalized_remaining_flips)

    p = softmax(logits)

    return p, v
end

function TS.batch_action_logprobs_and_values(policy, state)
    ets, econn, epairs, normalized_remaining_flips = state

    logits, vals = PV.eval_batch(policy, ets, econn, epairs, normalized_remaining_flips)
    logprobs = logsoftmax(logits, dims = 1)

    return logprobs, vals
end

function TS.initialize_state_data(env)
    return StateData(env)
end

function single_trajectory_return(env, policy)
    done = TS.is_terminal(env)
    if done
        return 0.0
    else
        initial_score = EF.score(env)
        minscore = initial_score
        while !done
            probs, val = TS.action_probabilities_and_value(policy, TS.state(env))
            action = rand(Categorical(probs))

            TS.step!(env, action)

            minscore = min(minscore, EF.score(env))
            done = TS.is_terminal(env)
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
        TS.reset!(env, nflips = env.num_initial_flips)
        ret[idx] = single_trajectory_normalized_return(env, policy)
    end
    return mean(ret), std(ret)
end

function returns_versus_nflips(policy, env, num_trajectories; maxflipfactor = 1.0)
    avg, dev = average_normalized_returns(env, policy, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t AVG RET = %1.3f\t STD DEV = %1.3f\n" env.num_initial_flips env.maxflips avg dev
    return avg
end

function single_trajectory_tree_return(
    env,
    policy,
    tree_exploration_factor,
    probability_weight,
    discount,
    maxiter,
    temperature,
)
    done = TS.is_terminal(env)
    if done
        return 0.0
    else
        initial_score = EF.score(env)
        minscore = initial_score

        p, v = TS.action_probabilities_and_value(policy, TS.state(env))
        root = TS.Node(p, v, TS.is_terminal(env))

        while !done
            probs = TS.tree_action_probabilities!(
                root,
                policy,
                env,
                tree_exploration_factor,
                probability_weight,
                discount,
                maxiter,
                temperature,
            )

            action = rand(Categorical(probs))

            TS.step!(env, action)
            root = TS.get_new_root(root, action, env, policy)

            minscore = min(minscore, EF.score(env))
            done = TS.is_terminal(env)
        end
        return initial_score - minscore
    end
end

function single_trajectory_normalized_tree_return(
    env,
    policy,
    tree_exploration_factor,
    probability_weight,
    discount,
    maxiter,
    temperature,
)
    maxreturn = EF.score(env) - env.optimum_score
    if maxreturn == 0
        return 1.0
    else
        ret = single_trajectory_tree_return(
            env,
            policy,
            tree_exploration_factor,
            probability_weight,
            discount,
            maxiter,
            temperature,
        )
        return ret / maxreturn
    end
end

function average_normalized_tree_returns(
    env,
    policy,
    tree_exploration_factor,
    probability_weight,
    discount,
    maxiter,
    temperature,
    num_trajectories,
)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        TS.reset!(env, nflips = env.num_initial_flips)
        ret[idx] = single_trajectory_normalized_tree_return(
            env,
            policy,
            tree_exploration_factor,
            probability_weight,
            discount,
            maxiter,
            temperature,
        )
    end
    return mean(ret), std(ret)
end

function tree_returns_versus_nflips(
    policy,
    Cpuct,
    discount,
    maxtime,
    temperature,
    nref,
    nflips,
    num_trajectories;
    maxflipfactor = 1.0,
)
    maxflips = ceil(Int, maxflipfactor * nflips)
    env = EdgeFlip.OrderedGameEnv(nref, nflips, maxflips = maxflips)
    avg, dev = average_normalized_tree_returns(
        env,
        policy,
        Cpuct,
        discount,
        maxtime,
        temperature,
        num_trajectories,
    )
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t AVG RET = %1.3f\t STD DEV = %1.3f\n" env.num_initial_flips env.maxflips avg dev
    return avg
end
