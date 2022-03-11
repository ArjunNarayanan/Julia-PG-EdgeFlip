module GreedyPolicy
using EdgeFlip
using Statistics

function step!(env::EdgeFlip.GameEnv, action)
    num_actions = EdgeFlip.number_of_actions(env)
    if 0 < action <= num_actions
        EdgeFlip.step!(env, action)
    else
        EdgeFlip.step!(env)
    end
end

function is_terminated(env::EdgeFlip.GameEnv)
    return EdgeFlip.is_terminated(env)
end

function reward(env::EdgeFlip.GameEnv)
    return EdgeFlip.reward(env)
end

function reset!(env::EdgeFlip.GameEnv)
    EdgeFlip.reset!(env)
end

function score(env::EdgeFlip.GameEnv)
    return EdgeFlip.score(env)
end

function greedy_action(env)
    return rand(greedy_actions(env))
end

function greedy_actions(env)
    num_actions = EdgeFlip.number_of_actions(env)
    rewards = [EdgeFlip.reward(env,e) for e in 1:num_actions]
    maxr = maximum(rewards)
    actions = findall(rewards .== maxr)
    
    return actions
end

function action_probabilities(greedy_actions,num_actions)
    num_greedy_actions = length(greedy_actions)
    @assert num_greedy_actions > 0
    probs = zeros(num_actions)
    probs[greedy_actions] .= 1.0/length(greedy_actions)
    return probs
end

function action_probabilities(env)
    num_actions = EdgeFlip.number_of_actions(env)
    ga = greedy_actions(env)
    probs = action_probabilities(ga, num_actions)
    return probs
end

function single_trajectory_return(env)
    ep_returns = []
    done = is_terminated(env)
    if done
        return 0.0
    else
        while !done
            action = greedy_action(env)
            step!(env, action)
            push!(ep_returns, reward(env))
            done = is_terminated(env)
        end
        return sum(ep_returns)
    end
end

function single_trajectory_normalized_return(env)
    maxscore = score(env)
    if maxscore == 0
        return 1.0
    else
        ret = single_trajectory_return(env)
        return ret/maxscore
    end
end

function normalized_returns(env, num_trajectories)
    ret = zeros(num_trajectories)
    for idx in 1:num_trajectories
        reset!(env)
        ret[idx] = single_trajectory_normalized_return(env)
    end
    return ret    
end


function average_returns(env, num_trajectories)
    ret = zeros(num_trajectories)
    for idx in 1:num_trajectories
        reset!(env)
        ret[idx] = single_trajectory_return(env)
    end
    return mean(ret)
end

function average_normalized_returns(env, num_trajectories)
    ret = zeros(num_trajectories)
    for idx in 1:num_trajectories
        reset!(env)
        ret[idx] = single_trajectory_normalized_return(env)
    end
    return mean(ret)
end

end