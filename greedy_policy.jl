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
    it,j = EdgeFlip.greedy_action(env)
    r = EdgeFlip.reward(env,(it,j))

    # edgeid = r < 0 ? 0 : env.mesh.t2e[it,j]
    edgeid = env.mesh.t2e[it,j]
    return edgeid
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
        return 0.0
    else
        ret = single_trajectory_return(env)
        return ret/maxscore
    end
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