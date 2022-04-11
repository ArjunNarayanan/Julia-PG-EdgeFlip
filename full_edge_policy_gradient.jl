module FullEdgePolicyGradient

using Flux
using Distributions: Categorical
using Printf
using Statistics
using BSON: @save


state(env) = nothing
step!(env, action) = nothing
reward(env) = nothing
is_terminated(env) = nothing
reset!(env) = nothing
score(env) = nothing
eval_single(policy, ets, econn) = nothing
eval_batch(policy, ets, econn) = nothing

function policy_gradient_loss(ets, econn, policy, actions, weights)
    logits = eval_batch(policy, ets, econn)
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

function idx_to_action(idx)
    triangle, vertex = div(idx - 1, 3) + 1, (idx - 1) % 3 + 1
    return triangle, vertex
end

function collect_batch_trajectories(env, policy, batch_size, discount)
    batch_ets = []
    batch_econn = []
    batch_actions = []
    ep_rewards = []
    batch_weights = []
    batch_returns = []

    counter = 0
    reset!(env)

    while counter < batch_size
        ets, econn = state(env)
        logits = vec(eval_single(policy, ets, econn))
        probs = Categorical(softmax(logits))
        action_index = rand(probs)

        action = idx_to_action(action_index)

        step!(env, action)
        r = reward(env)
        done = is_terminated(env)

        push!(batch_ets, ets)

        num_actions = size(ets, 2)
        push!(batch_econn, econn)

        append!(batch_actions, action_index)
        append!(ep_rewards, r)

        counter += 1

        if done || counter >= batch_size
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

    batch_ets = cat(batch_ets..., dims = 3)
    batch_econn = cat(batch_econn..., dims = 2)

    states = (batch_ets, batch_econn)

    avg_return = sum(batch_returns) / length(batch_returns)

    return states, batch_actions, batch_weights, avg_return
end

function step_epoch(env, policy, optimizer, batch_size, discount)
    states, actions, weights, avg_return =
        collect_batch_trajectories(env, policy, batch_size, discount)

    θ = params(policy)
    loss = 0.0
    grads = Flux.gradient(θ) do
        loss = policy_gradient_loss(states[1], states[2], policy, actions, weights)
    end

    Flux.update!(optimizer, θ, grads)
    return loss, avg_return
end

function single_trajectory_return(env, policy)
    done = is_terminated(env)
    if done
        return 0.0
    else
        initial_score = score(env)
        minscore = initial_score
        while !done
            ets, econn = state(env)
            probs = softmax(vec(eval_single(policy, ets, econn)))
            action_index = rand(Categorical(probs))

            action = idx_to_action(action_index)
            step!(env, action)

            minscore = min(minscore, score(env))
            done = is_terminated(env)
        end
        return initial_score - minscore
    end
end

function single_trajectory_normalized_return(env, policy)
    maxreturn = score(env) - env.optimum_score
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
        reset!(env, nflips = env.num_initial_flips)
        ret[idx] = single_trajectory_normalized_return(env, policy)
    end
    return mean(ret)
end

function train_and_save_best_models(
    env,
    policy,
    optimizer,
    batch_size,
    discount,
    num_epochs,
    evaluator;
    evaluate_every = 500,
    foldername = "results/models/new-edge-model/",
    generate_plots = true,
)

    ret = evaluator(policy)

    for epoch = 1:num_epochs
        loss, avg_return = step_epoch(env, policy, optimizer, batch_size, discount)

        if epoch % evaluate_every == 0
            new_rets = evaluator(policy)

            old_err = sum((1.0 .- ret) .^ 2)
            new_err = sum((1.0 .- new_rets) .^ 2)

            if new_err < old_err

                if generate_plots
                    plot_filename = foldername * "policy-" * string(epoch) * ".png"
                    plot_returns(new_rets, filename = plot_filename)
                end

                model_filename = foldername * "policy-" * string(epoch) * ".bson"
                @save model_filename policy new_rets

                ret .= new_rets

                average_return = sum(new_rets) / length(new_rets)

                statement = @sprintf "epoch: %3d \t avg return: %3.2f" epoch average_return
                println(statement)
            end
        end
    end
end


end
