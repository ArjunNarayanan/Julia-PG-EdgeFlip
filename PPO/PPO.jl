module PPO

using Printf
using Distributions: Categorical
using Flux
using Random

function not_implemented(name)
    error("Function $name needs to be overloaded")
end

function state(env) not_implemented("state") end
function reward(env) not_implemented("reward") end
function is_terminal(env) not_implemented("is_terminal") end
function reset!(env) not_implemented("reset!") end
function step!(env, action) not_implemented("step!") end

function initialize_state_data(env) not_implemented("initialize_state_data") end
function update!(state_data, state) not_implemented("update!") end
function action_probabilities(policy, state) not_implemented("action_probabilities") end
function batch_action_probabilities(policy, state) not_implemented("batch_action_probabilities") end
function episode_state(state_data) not_implemented("episode_state") end
function episode_returns(rewards, state_data, discount) not_implemented("episode_returns") end
function batch_state(state_data) not_implemented("batch_state") end
function batch_advantage(episodes) not_implemented("batch_advantage") end


##############################################################################################################################

struct EpisodeData
    state_data::Any
    selected_action_probabilities::Any
    selected_actions::Any
    rewards::Any
end

function EpisodeData(state_data)
    selected_action_probabilities = Float64[]
    selected_actions = Int64[]
    rewards = Float64[]
    EpisodeData(state_data, selected_action_probabilities, selected_actions, rewards)
end

function selected_actions(data::EpisodeData)
    return data.selected_actions
end

function rewards(data::EpisodeData)
    return data.rewards
end

function selected_action_probabilities(data::EpisodeData)
    return data.selected_action_probabilities
end

function state_data(data::EpisodeData)
    return data.state_data
end

function update!(episode::EpisodeData, state, action_probability, action, reward)
    update!(episode.state_data, state)
    push!(episode.selected_action_probabilities, action_probability)
    push!(episode.selected_actions, action)
    push!(episode.rewards, reward)
    return
end

function Base.length(b::EpisodeData)
    @assert length(b.selected_action_probabilities) ==
            length(b.selected_actions) ==
            length(b.rewards)
    return length(b.selected_action_probabilities)
end

function Base.show(io::IO, data::EpisodeData)
    nd = length(data)
    println(io, "EpisodeData\n\t$nd data points")
end

function collect_step_data!(episode_data, env, policy)
    s = state(env)
    ap = action_probabilities(policy, s)
    a = rand(Categorical(ap))

    step!(env, a)
    r = reward(env)

    update!(episode_data, s, ap[a], a, r)
end

function collect_episode_data!(episode_data, env, policy)
    terminal = is_terminal(env)

    while !terminal
        collect_step_data!(episode_data, env, policy)
        terminal = is_terminal(env)
    end
end

function batch_episode(episode::EpisodeData, discount)
    s = episode_state(state_data(episode))
    r = episode_returns(rewards(episode), state_data(episode), discount)
    return EpisodeData(
        s,
        selected_action_probabilities(episode),
        selected_actions(episode),
        r,
    )
end
##############################################################################################################################




##############################################################################################################################
struct Rollouts
    episodes::Any
end

function Rollouts()
    Rollouts(EpisodeData[])
end

function Base.length(b::Rollouts)
    return length(b.episodes)
end

function Base.show(io::IO, batch_data::Rollouts)
    num_episodes = length(batch_data)
    println(io, "Rollouts\n\t$num_episodes episodes")
end

function Base.getindex(b::Rollouts, idx)
    return b.episodes[idx]
end

function Random.shuffle!(b::Rollouts)
    shuffle!(b.episodes)
end

function update!(b::Rollouts, episode)
    push!(b.episodes, episode)
end

function collect_rollouts!(rollouts, env, policy, discount, num_episodes)
    while length(rollouts) < num_episodes
        reset!(env)
        if !is_terminal(env)
            episode = EpisodeData(initialize_state_data(env))
            collect_episode_data!(episode, env, policy)
            
            episode = batch_episode(episode, discount)
            update!(rollouts, episode)
        end
    end
end

function simplified_ppo_clip(epsilon, advantage)
    return [a >= 0 ? (1 + epsilon) * a : (1 - epsilon) * a for a in advantage]
end

function ppo_loss(policy, state, actions, old_action_probabilities, advantage, epsilon)
    ap = batch_action_probabilities(policy, state)
    selected_ap = [ap[a, idx] for (idx, a) in enumerate(actions)]

    ppo_gain = @. selected_ap / old_action_probabilities * advantage
    ppo_clip = simplified_ppo_clip(epsilon, advantage)

    loss = -Flux.mean(min.(ppo_gain, ppo_clip))

    return loss
end

function batch_selected_action_probabilities(episodes)
    return cat(selected_action_probabilities.(episodes)..., dims = 1)
end

function batch_selected_actions(episodes)
    return cat(selected_actions.(episodes)..., dims = 1)
end

function step_batch!(policy, optimizer, episodes, epsilon)
    state = batch_state(state_data.(episodes))
    old_action_probabilities = batch_selected_action_probabilities(episodes)
    advantage = batch_advantage(episodes)
    sel_actions = batch_selected_actions(episodes)

    weights = Flux.params(policy)

    loss = 0.0
    grads = Flux.gradient(weights) do
        loss = ppo_loss(
            policy,
            state,
            sel_actions,
            old_action_probabilities,
            advantage,
            epsilon,
        )
    end

    Flux.update!(optimizer, weights, grads)

    return loss
end

function step_epoch!(policy, optimizer, batch_data, epsilon, batch_size)
    num_episodes = length(batch_data)
    start = 1
    loss = []
    while start < num_episodes
        stop = min(start + batch_size, num_episodes)
        episodes = batch_data[start:stop]

        l = step_batch!(policy, optimizer, episodes, epsilon)
        push!(loss, l)

        start += batch_size
    end
    return Flux.mean(loss)
end

function ppo_train!(
    policy,
    optimizer,
    batch_data,
    epsilon,
    batch_size,
    num_epochs,
)
    for epoch = 1:num_epochs
        shuffle!(batch_data)
        l = step_epoch!(policy, optimizer, batch_data, epsilon, batch_size)
        @printf "EPOCH : %d \t AVG LOSS : %1.4f\n" epoch l
    end
end

function ppo_iterate!(
    policy,
    env,
    optimizer,
    episodes_per_iteration,
    discount,
    epsilon,
    batch_size,
    num_epochs,
    num_iter,
    evaluator
)
    for iter in 1:num_iter
        println("\nPPO ITERATION : $iter")

        rollouts = Rollouts()
        collect_rollouts!(rollouts, env, policy, discount, episodes_per_iteration)

        ppo_train!(policy, optimizer, rollouts, epsilon, batch_size, num_epochs)

        ret, dev = evaluator(policy, env)

        @printf "RET = %1.4f\tDEV = %1.4f\n" ret dev
    end
end

end
