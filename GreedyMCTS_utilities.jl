using Statistics
using EdgeFlip

include("GreedyMCTS.jl")
include("greedy_policy.jl")
GTS = GreedyMCTS
GP = GreedyPolicy
EF = EdgeFlip

function GTS.is_terminal(env::EF.GameEnv)
    return EF.done(env)
end

function GTS.step!(env, action)
    EF.step!(env, action)
end

function GTS.reverse_step!(env, action)
    EF.reverse_step!(env, action)
end

function GTS.number_of_actions(env::EF.GameEnv)
    return EF.number_of_actions(env)
end

function single_trajectory_greedy_returns_and_actions(env)
    initial_score = EF.score(env)
    done = GTS.is_terminal(env)

    minscore = initial_score
    actions = Int[]

    while !done
        action = GP.greedy_action(env)
        push!(actions, action)
        GTS.step!(env, action)
        minscore = min(minscore, EF.score(env))
        done = GTS.is_terminal(env)
    end
    ret = initial_score - minscore
    return ret, actions
end

function GTS.estimate_value(env; num_traces = 1)
    value = 0.0
    for trace = 1:num_traces
        maxreturn = EF.score(env) - EF.optimum_score(env)
        if maxreturn == 0
            value += 1.0
        else
            ret, actions = single_trajectory_greedy_returns_and_actions(env)
            value += ret / maxreturn
            reverse_actions!(env, actions)
        end
    end
    return value / num_traces
end

function reverse_actions!(env, actions)
    for action in reverse(actions)
        GTS.reverse_step!(env, action)
    end
end

function single_trajectory_tree_return(env, tree_settings)
    initial_score = EF.score(env)
    minscore = initial_score

    root = GTS.Node(env)
    done = GTS.is_terminal(env)

    while !done
        root = GTS.step_mcts!(root, env, tree_settings)
        done = GTS.is_terminal(env)
        minscore = min(minscore, EF.score(env))
    end

    return initial_score - minscore
end

function single_trajectory_normalized_tree_return(env, tree_settings)
    maxreturn = EF.score(env) - EF.optimum_score(env)
    if maxreturn == 0
        return 1.0
    else
        ret = single_trajectory_tree_return(env, tree_settings)
        return ret / maxreturn
    end
end

function average_normalized_tree_returns(env, tree_settings, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        EF.reset!(env, nflips = env.num_initial_flips, maxflips = env.maxflips)
        ret[idx] = single_trajectory_normalized_tree_return(env, tree_settings)
    end
    return mean(ret), std(ret)
end