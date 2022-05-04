module PPO

using Distributions: Categorical

function state(env) end
function reward(env) end
function is_terminal(env) end
function reset!(env) end

function update!(state_data, state) end
function action_probabilities(policy, state) end

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

function update!(batch_data::BatchData, state, action_probability, action, reward, terminal)
    update!(batch_data.state_data, state)
    push!(batch_data.action_probabilities, action_probability)
    push!(batch_data.actions, action)
    push!(batch_data.rewards, reward)
    push!(batch_data.terminal, terminal)
end

function Base.length(b::BatchData)
    @assert length(b.action_probabilities) == length(b.actions) == length(b.rewards) == length(b.terminal)
    return length(b.action_probabilities)
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
        collect_sample_trajectory!(batch_data, env, policy)
    end
end

end