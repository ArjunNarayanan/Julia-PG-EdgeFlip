using Statistics
using EdgeFlip

include("GreedyMCTS.jl")
include("greedy_policy.jl")
GTS = GreedyMCTS
GP = GreedyPolicy

function GTS.is_terminal(env::EdgeFlip.GameEnv)
    return EdgeFlip.done(env)
end

function GTS.step!(env, action)
    EdgeFlip.step!(env, action)
end

function GTS.reverse_step!(env, action)
    EdgeFlip.reverse_step!(env, action)
end

function single_trajectory_returns_and_actions(env)
    done = GTS.is_terminal(env)
    if done
        return 0.0
    else
        initial_score = EdgeFlip.score(env)
        minscore = initial_score
        actions = Int[]
        while !done
            action = GP.greedy_action(env)
            push!(actions, action)
            GTS.step!(env, action)
            minscore = min(minscore, EdgeFlip.score(env))
            done = GTS.is_terminal(env)
        end
        ret = initial_score - minscore
        return ret, actions
    end
end

function GTS.estimate_value(env; num_traces = 1)
    value = 0.0
    for trace in 1:num_traces
        maxreturn = EdgeFlip.score(env) - EdgeFlip.optimum_score(env)
        if maxreturn == 0
            value += 1.0
        else
            ret, actions = single_trajectory_returns_and_actions(env)
            value += ret/maxreturn
            reverse_actions!(env, actions)
        end
    end
    return value/num_traces
end

function reverse_actions!(env, actions)
    for action in reverse(actions)
        GTS.reverse_step!(env, action)
    end
end