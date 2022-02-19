module PolicyGradient

# using EdgeFlip: state, step!, reward, is_terminated, reset!, score
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

function policy_gradient_loss(states, policy, actions, weights)
    logp = logsoftmax.(policy.(states))
    logp = -[logp[i][actions[i]] for i = 1:length(actions)]
    loss = Flux.mean(logp .* weights)
    return loss
end

function step_trajectory(env, policy)
    logits = policy(state(env))
    probs = Categorical(softmax(logits))
    action = rand(probs)

    logp = logsoftmax(logits)[action]
    r = reward(env)
    return logp, r
end

function collect_batch_trajectories(env, policy, batch_size)
    batch_states = []
    batch_actions = []
    ep_rewards = []
    batch_weights = []
    batch_returns = []

    while true
        s = state(env)
        logits = policy(s)
        probs = Categorical(softmax(logits))
        action = rand(probs)

        step!(env, action)
        r = reward(env)
        done = is_terminated(env)

        push!(batch_states, s)
        append!(batch_actions, action)
        append!(ep_rewards, r)

        if done || length(batch_actions) >= batch_size
            ep_ret = cumsum(reverse(ep_rewards))
            append!(batch_weights, ep_ret)
            # ep_ret, ep_len = sum(ep_rewards), length(ep_rewards)
            # append!(batch_weights, repeat([ep_ret], ep_len))

            append!(batch_returns, ep_ret)

            if length(batch_actions) >= batch_size
                break
            else
                reset!(env)
                ep_rewards = []
            end
        end
    end

    avg_return = sum(batch_returns) / length(batch_returns)

    return batch_states, batch_actions, batch_weights, avg_return
end

function step_epoch(env, policy, optimizer, batch_size)
    states, actions, weights, avg_return =
        collect_batch_trajectories(env, policy, batch_size)

    θ = params(policy)
    loss = 0.0
    grads = Flux.gradient(θ) do
        loss = policy_gradient_loss(states, policy, actions, weights)
    end

    Flux.update!(optimizer, θ, grads)
    return loss, avg_return
end


function single_trajectory_normalized_return(env, policy, maxsteps)
    reset!(env)
    maxscore = score(env)
    ep_returns = []
    counter = 1
    done = is_terminated(env)
    if done
        return 1.0
    else
        while !done && counter <= maxsteps
            action = env |> state |> policy |> softmax |> Categorical |> rand
            step!(env, action)
            push!(ep_returns, reward(env))
            done = is_terminated(env)
            counter += 1
        end
        return sum(ep_returns) / maxscore
    end
end

function average_returns(env, policy, maxsteps, num_trajectories)
    ret = [
        single_trajectory_normalized_return(env, policy, maxsteps) for
        i = 1:num_trajectories
    ]
    return sum(ret) / length(ret)
end

function mean_and_std_returns(env, policy, maxsteps, num_trajectories)
    ret = [
        single_trajectory_normalized_return(env, policy, maxsteps) for
        i = 1:num_trajectories
    ]
    avg = mean(ret)
    dev = std(ret)
    return avg, dev
end

function run_training_loop(
    env,
    policy,
    batch_size,
    num_epochs,
    learning_rate,
    maxsteps,
    num_trajectories;
    estimate_every = 10,
)
    optimizer = ADAM(learning_rate)
    return_history = []
    epoch_history = []

    for epoch = 1:num_epochs
        loss, avg_return = step_epoch(env, policy, optimizer, batch_size)

        if epoch % estimate_every == 0
            avg_return = average_returns(env, policy, maxsteps, num_trajectories)
            append!(return_history, avg_return)
            append!(epoch_history, epoch)
            statement =
                @sprintf "epoch: %3d \t loss: %.4f \t avg return: %3.2f" epoch loss avg_return
            println(statement)
        end
    end
    return epoch_history, return_history
end



end