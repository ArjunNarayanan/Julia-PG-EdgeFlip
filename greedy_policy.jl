module GreedyPolicy
using EdgeFlip
using EdgeFlip: state, step!, reward, is_terminated, reset!, score
using Flux
using Distributions: Categorical
using Statistics

function greedy_action(env)
    num_actions = EdgeFlip.number_of_edges(env.mesh)
    it,j = EdgeFlip.greedy_action(env)
    action = env.mesh.t2e[it,j]
    return action
end

function single_trajectory_normalized_return(env, maxsteps)
    reset!(env)
    maxscore = score(env)
    ep_returns = []
    counter = 1
    done = is_terminated(env)
    while !done && counter <= maxsteps
        action = greedy_action(env)
        step!(env, action)
        push!(ep_returns, reward(env))
        done = is_terminated(env)
        counter += 1
    end
    return sum(ep_returns)/maxscore
end

function mean_and_std_returns(env,maxsteps,num_trajectories)
    ret =
        [single_trajectory_normalized_return(env, maxsteps) for i = 1:num_trajectories]
    avg = mean(ret)
    dev = std(ret)
    return avg, dev
end

end