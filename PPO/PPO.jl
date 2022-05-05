module PPO

using Distributions: Categorical
using Flux

function state(env) end
function reward(env) end
function is_terminal(env) end
function reset!(env) end
function step!(env, action) end

function initialize_state_data(env) end
function update!(state_data, state) end
function action_probabilities(policy, state) end
function batch_action_probabilities(policy, state) end
function batch_state(state_data) end

struct BatchData
    state_data
    action_probabilities
    actions
    rewards
    terminal
    function BatchData(state_data)
        action_probabilities = Float64[]
        actions = Int64[]
        rewards = Float64[]
        terminal = Bool[]
        new(state_data, action_probabilities, actions, rewards, terminal)
    end
end

function actions(data::BatchData)
    return data.actions
end

function rewards(data::BatchData)
    return data.rewards
end

function terminal(data::BatchData)
    return data.terminal
end

function action_probabilities(data::BatchData)
    return data.action_probabilities
end

function state_data(data::BatchData)
    return data.state_data
end

function update!(batch_data::BatchData, state, action_probability, action, reward, terminal)
    update!(batch_data.state_data, state)
    push!(batch_data.action_probabilities, action_probability)
    push!(batch_data.actions, action)
    push!(batch_data.rewards, reward)
    push!(batch_data.terminal, terminal)
    return
end

function Base.length(b::BatchData)
    @assert length(b.action_probabilities) == length(b.actions) == length(b.rewards) == length(b.terminal)
    return length(b.action_probabilities)
end

function Base.show(io::IO, data::BatchData)
    nd = length(data)
    println(io, "BatchData\n\t$nd data points")
end

function collect_step_data!(batch_data, env, policy)
    s = state(env)
    ap = action_probabilities(policy, s)
    a = rand(Categorical(ap))

    step!(env, a)
    r = reward(env)
    t = is_terminal(env)

    update!(batch_data, s, ap[a], a, r, t)
end

function collect_sample_trajectory!(batch_data, env, policy)
    terminal = is_terminal(env)

    while !terminal
        collect_step_data!(batch_data, env, policy)
        terminal = is_terminal(env)
    end
end

function collect_batch_data!(batch_data, env, policy, memory_size)
    while length(batch_data) < memory_size
        reset!(env)
        collect_sample_trajectory!(batch_data, env, policy)
    end
end

function batch_returns(rewards, terminal, discount)
    l = length(rewards)
    @assert l == length(terminal)

    values = zeros(l)
    counter = 0
    v = 0.0

    for idx = l:-1:1
        if terminal[idx]
            counter = 0
            v = 0.0
        end
        v = rewards[idx] + discount * v
        values[idx] = v
        counter += 1
    end

    return values
end

function batch_returns(data::BatchData, discount)
    return batch_returns(data.rewards, data.terminal, discount)
end

function simplified_ppo_clip(epsilon, advantage)
    return [a >= 0 ? (1+epsilon)*a : (1-epsilon)*a for a in advantage]
end

function ppo_loss(policy, state, actions, old_action_probabilities, advantage, epsilon)
    ap = batch_action_probabilities(policy, state)
    selected_ap = [ap[a,idx] for (idx,a) in enumerate(actions)]

    ppo_gain = @. selected_ap / old_action_probabilities * advantage
    ppo_clip = simplified_ppo_clip(epsilon, advantage)

    loss = -Flux.mean(min.(ppo_gain, ppo_clip))

    return loss
end

function step_batch!(policy, optimizer, data, discount, epsilon)
    state = batch_state(state_data(data))
    old_action_probabilities = action_probabilities(data)
    advantage = batch_returns(data, discount)
    a = actions(data)

    weights = params(policy)

    loss = 0.0
    grads = Flux.gradient(weights) do 
        loss = ppo_loss(policy, state, a, old_action_probabilities, advantage, epsilon)
    end

    Flux.update!(optimizer, weights, grads)

    return loss
end


end