module Supervised

using Flux
using Printf
using EdgeFlip

state(env) = nothing

function collect_training_data!(state_data, greedy_probs, env)
    num_features, num_actions, num_batches = size(state_data)
    @assert size(greedy_probs) == (num_actions, num_batches)
    counter = 1
    while counter <= num_batches
        EdgeFlip.reset!(env)
        done = EdgeFlip.is_terminated(env)
        if !done
            s = state(env)
            state_data[:, :, counter] .= s
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

function run_training_loop(env, policy, batch_size, num_epochs, learning_rate; print_every = 100)
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
            logits = reshape(policy(state_data),na,batch_size)
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