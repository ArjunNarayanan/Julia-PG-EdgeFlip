module EdgePolicyGradient

using Flux
using Distributions: Categorical
using Printf
using Statistics


state(env) = nothing
step!(env, action) = nothing
reward(env) = nothing
is_terminated(env) = nothing
reset!(env) = nothing
score(env) = nothing

function policy_gradient_loss(
    states,
    policy,
    actions,
    weights,
)
    num_batches = length(actions)
    logits = policy(states, num_batches)
    logp = logsoftmax(logits, dims = 2)
    selected_logp = -[logp[1, action, idx] for (idx, action) in enumerate(actions)]
    loss = Flux.mean(selected_logp .* weights)
    return loss
end

function advantage(rewards, discount)
    numsteps = length(rewards)
    weights = zeros(numsteps)
    weights[numsteps] = rewards[numsteps]
    for step = (numsteps-1):-1:1
        weights[step] = rewards[step] + discount * weights[step+1]
    end
    return weights
end

function collect_batch_trajectories(env, policy, batch_size, discount)
    batch_vertex_scores = []
    batch_edge_templates = []
    batch_actions = []
    ep_rewards = []
    batch_weights = []
    batch_returns = []

    while true
        s = state(env)
        logits = vec(policy(s))
        probs = Categorical(softmax(logits))
        action = rand(probs)

        step!(env, action)
        r = reward(env)
        done = is_terminated(env)

        vs, et = s
        push!(batch_vertex_scores, vs)
        push!(batch_edge_templates, et)
        append!(batch_actions, action)
        append!(ep_rewards, r)

        if done || length(batch_actions) >= batch_size
            ep_ret = advantage(ep_rewards, discount)
            append!(batch_weights, ep_ret)
            append!(batch_returns, sum(ep_rewards))

            if length(batch_actions) >= batch_size
                break
            else
                reset!(env)
                ep_rewards = []
            end
        end
    end

    mvs = cat(batch_vertex_scores..., dims = 3)
    met = cat(batch_edge_templates..., dims = 3)

    states = (mvs, met)

    avg_return = sum(batch_returns) / length(batch_returns)

    return states, batch_actions, batch_weights, avg_return
end

function step_epoch(env, policy, optimizer, batch_size, discount)
    states, actions, weights, avg_return =
        collect_batch_trajectories(env, policy, batch_size, discount)

    θ = params(policy)
    loss = 0.0
    grads = Flux.gradient(θ) do
        loss = policy_gradient_loss(states, policy, actions, weights)
    end

    Flux.update!(optimizer, θ, grads)
    return loss, avg_return
end

function single_trajectory_return(env, policy)
    ep_returns = []
    done = is_terminated(env)
    if done
        return 0.0
    else
        while !done
            probs = vec(softmax(policy(state(env)), dims = 2))
            action = rand(Categorical(probs))
            step!(env, action)
            push!(ep_returns, reward(env))
            done = is_terminated(env)
        end
        return sum(ep_returns)
    end
end

function single_trajectory_normalized_return(env, policy)
    maxscore = score(env)
    if maxscore == 0
        return 0.0
    else
        ret = single_trajectory_return(env, policy)
        return ret / maxscore
    end
end

function average_returns(env, policy, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        reset!(env)
        ret[idx] = single_trajectory_return(env, policy)
    end
    return mean(ret)
end

function average_normalized_returns(env, policy, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        reset!(env)
        ret[idx] = single_trajectory_normalized_return(env, policy)
    end
    return mean(ret)
end

function run_training_loop(
    env,
    policy,
    batch_size,
    discount,
    num_epochs,
    learning_rate;
    print_every = 100,
)
    optimizer = ADAM(learning_rate)
    return_history = []
    epoch_history = []

    for epoch = 1:num_epochs
        loss, avg_return = step_epoch(env, policy, optimizer, batch_size, discount)

        append!(return_history, avg_return)
        append!(epoch_history, epoch)

        if epoch % print_every == 0
            statement =
            @sprintf "epoch: %3d \t loss: %.4f \t avg return: %3.2f" epoch loss avg_return
            println(statement)
        end
    end
    return epoch_history, return_history
end

function single_trajectory_return(env, policy)
    ep_returns = []
    done = is_terminated(env)
    if done
        return 0.0
    else
        while !done
            probs = vec(softmax(policy(state(env)), dims = 2))
            action = rand(Categorical(probs))
            step!(env, action)
            push!(ep_returns, reward(env))
            done = is_terminated(env)
        end
        return sum(ep_returns)
    end
end

function single_trajectory_normalized_return(env, policy)
    maxscore = score(env)
    if maxscore == 0
        return 0.0
    else
        ret = single_trajectory_return(env, policy)
        return ret / maxscore
    end
end

function average_returns(env, policy, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        reset!(env)
        ret[idx] = single_trajectory_return(env, policy)
    end
    return mean(ret)
end

function average_normalized_returns(env, policy, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        reset!(env, nflips = env.num_initial_flips)
        ret[idx] = single_trajectory_normalized_return(env, policy)
    end
    return mean(ret)
end


end