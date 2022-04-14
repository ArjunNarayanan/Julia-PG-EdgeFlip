module Supervised

using Flux
using Printf
using EdgeFlip

state(env) = nothing
reset!(env) = nothing

function collect_training_data!(state_data, greedy_probs, env)
    num_features, num_actions, num_batches = size(state_data)
    @assert size(greedy_probs) == (num_actions, num_batches)
    counter = 1
    while counter <= num_batches
        reset!(env)
        done = EdgeFlip.is_terminated(env)
        if !done
            s = state(env)
            state_data[:, :, counter] .= s
            greedy_probs[:, counter] .= greedy_action_distribution(env)
            counter += 1
        end
    end
end

function collect_edge_training_data!(
    edge_template_score,
    edge_connectivity,
    edge_pairs,
    greedy_probs,
    env,
)
    _, num_actions, num_batches = size(edge_template_score)
    @assert length(edge_connectivity) == 3num_actions
    @assert size(edge_pairs) == (num_actions, num_batches)
    @assert size(greedy_probs) == (num_actions, num_batches)
    
    counter = 1
    edge_connectivity .= env.edge_connectivity
    while counter <= num_batches
        reset!(env)
        done = EdgeFlip.is_terminated(env)
        if !done
            ets, econn, epairs = state(env)
            edge_template_score[:, :, counter] .= ets
            edge_pairs[:, counter] .= epairs
            greedy_probs[:, counter] .= greedy_action_distribution(env)
            counter += 1
        end
    end
end

function greedy_action_distribution(env)
    na = EdgeFlip.number_of_actions(env)
    actions = 1:na
    rewards = [EdgeFlip.reward(env, a) for a in actions]
    probs = zeros(na)
    maxindices = findall(rewards .== maximum(rewards))
    probs[maxindices] .= 1.0 / (length(maxindices))
    return probs
end

function run_training_loop(
    env,
    policy,
    batch_size,
    num_epochs,
    learning_rate;
    print_every = 100,
)
    loss_history = zeros(num_epochs)

    s = state(env)
    nf, na = size(s)
    state_data = zeros(nf, na, batch_size)
    greedy_probs = zeros(na, batch_size)

    optimizer = ADAM(learning_rate)

    for epoch = 1:num_epochs
        collect_training_data!(state_data, greedy_probs, env)

        theta = Flux.params(policy)
        loss = 0.0
        grads = Flux.gradient(theta) do
            logits = reshape(policy(state_data), na, batch_size)
            loss = Flux.logitcrossentropy(logits, greedy_probs)
        end

        Flux.update!(optimizer, theta, grads)
        loss_history[epoch] = loss

        if epoch % print_every == 0
            @printf "epoch: %3d \t loss: %.3e\n" epoch loss
        end
    end
    return loss_history
end

function run_edge_training_loop(
    env,
    policy,
    batch_size,
    num_epochs,
    learning_rate;
    print_every = 100,
)
    loss_history = zeros(num_epochs)

    na = EdgeFlip.number_of_actions(env)

    edge_template_score = zeros(Int, 4, na, batch_size)
    edge_connectivity = zeros(Int, na)
    edge_pairs = zeros(Int, na, batch_size)
    greedy_probs = zeros(na, batch_size)

    optimizer = ADAM(learning_rate)

    for epoch = 1:num_epochs
        collect_edge_training_data!(edge_template_score, edge_connectivity, edge_pairs, greedy_probs, env)

        theta = Flux.params(policy)
        loss = 0.0

        loss = 0.0
        grads = Flux.gradient(theta) do
            logits = eval_batch(policy, edge_template_score, edge_connectivity, edge_pairs)
            logits = reshape(logits, na, batch_size)
            loss = Flux.logitcrossentropy(logits, greedy_probs)
        end

        Flux.update!(optimizer, theta, grads)
        loss_history[epoch] = loss

        if epoch % print_every == 0
            @printf "epoch: %3d \t loss: %.3e\n" epoch loss
        end
    end
    return loss_history
end

end